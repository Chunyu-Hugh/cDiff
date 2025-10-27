import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable
import math

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class SBIPosteriorNetwork(nn.Module):
    """
    SBI后验网络 - 基于条件流的神经网络
    输入：观测数据x和归一化常数norm
    输出：参数的后验分布估计
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, num_layers: int = 6,
                 context_dim: int = 2, output_dim: int = 6, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.output_dim = output_dim

        # 数据编码器（类似PointNet处理事件数据）
        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 上下文编码器（归一化常数）
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # 聚合编码器
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=10000)

        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=min(8, hidden_dim // 32),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头（均值和方差预测）
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 观测数据 (B, N, input_dim)
            context: 上下文信息 (B, context_dim)
        Returns:
            mean: 均值预测 (B, output_dim)
            logvar: 对数方差预测 (B, output_dim)
        """
        B, N, D = x.shape

        # 编码观测数据
        x_encoded = self.data_encoder(x.view(B * N, D)).view(B, N, -1)
        x_encoded = self.pos_encoder(x_encoded)

        # 编码上下文信息
        context_encoded = self.context_encoder(context)  # (B, hidden_dim//2)

        # Transformer处理序列数据
        transformer_out = self.transformer(x_encoded)  # (B, N, hidden_dim)

        # 全局池化
        global_feat = torch.max(transformer_out, dim=1).values  # (B, hidden_dim)

        # 聚合特征
        combined = torch.cat([global_feat, context_encoded], dim=1)  # (B, hidden_dim + hidden_dim//2)
        aggregated = self.aggregator(combined)  # (B, hidden_dim)

        # 预测均值和方差
        mean = self.mean_head(aggregated)
        logvar = self.logvar_head(aggregated)

        # 数值稳定性
        logvar = torch.clamp(logvar, min=-10, max=3)

        return mean, logvar

    def sample(self, x: torch.Tensor, context: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        从后验分布采样，确保采样的点在圆上
        Args:
            x: 观测数据 (B, N, input_dim)
            context: 上下文信息 (B, context_dim)
            num_samples: 采样数量
        Returns:
            samples: 采样结果 (num_samples, B, output_dim)
        """
        mean, logvar = self.forward(x, context)
        std = torch.exp(0.5 * logvar)

        # 重参数化技巧
        eps = torch.randn(num_samples, *mean.shape, device=mean.device)
        raw_samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        
        # 确保采样的点在圆上：将点投影到圆上
        B = mean.shape[0]
        radius = context.squeeze(-1)  # (B,) - 圆的半径
        
        # 计算每个采样点到圆心的距离
        distances = torch.sqrt(raw_samples[:, :, 0]**2 + raw_samples[:, :, 1]**2)  # (num_samples, B)
        
        # 将点投影到圆上：保持角度不变，调整距离到目标半径
        angles = torch.atan2(raw_samples[:, :, 1], raw_samples[:, :, 0])  # (num_samples, B)
        
        # 重新计算坐标，确保在圆上
        projected_x = radius.unsqueeze(0) * torch.cos(angles)  # (num_samples, B)
        projected_y = radius.unsqueeze(0) * torch.sin(angles)  # (num_samples, B)
        
        # 组合结果
        samples = torch.stack([projected_x, projected_y], dim=-1)  # (num_samples, B, 2)
        
        return samples

    def log_prob(self, x: torch.Tensor, context: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        计算参数的对数概率密度
        Args:
            x: 观测数据 (B, N, input_dim)
            context: 上下文信息 (B, context_dim)
            params: 参数样本 (B, output_dim)
        Returns:
            log_prob: 对数概率 (B,)
        """
        mean, logvar = self.forward(x, context)
        std = torch.exp(0.5 * logvar)

        # 高斯对数概率
        var = std ** 2
        log_prob = -0.5 * torch.sum(
            torch.log(2 * torch.pi * var) + ((params - mean).pow(2) / var),
            dim=1
        )

        return log_prob


class SBIModel(nn.Module):
    """
    完整的SBI模型，包含后验网络和训练逻辑
    """

    def __init__(self, config: dict):
        super().__init__()
        self.posterior_net = SBIPosteriorNetwork(
            input_dim=config.get('input_dim', 2),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            context_dim=config.get('context_dim', 2),
            output_dim=config.get('output_dim', 6),
            dropout=config.get('dropout', 0.1)
        )

        self.rounds = 0
        self.training_history = []

    def train_round(self, data_loader, optimizer, num_epochs: int = 100,
                   device: str = 'cpu') -> dict:
        """
        单轮SBI训练
        """
        self.train()
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_x, batch_context, batch_params in data_loader:
                batch_x = batch_x.to(device)
                batch_context = batch_context.to(device)
                batch_params = batch_params.to(device)

                optimizer.zero_grad()

                # 计算负对数似然损失
                log_probs = self.posterior_net.log_prob(batch_x, batch_context, batch_params)
                nll_loss = -log_probs.mean()
                
                # 添加几何约束损失：确保预测的点在圆上
                mean, logvar = self.posterior_net.forward(batch_x, batch_context)
                # 计算预测点到圆心的距离
                predicted_radius = torch.sqrt(mean[:, 0]**2 + mean[:, 1]**2)
                # 从context中获取真实的z值（圆的半径）
                true_radius = batch_context.squeeze(-1)  # (B,)
                # 几何约束损失：预测半径应该等于真实半径
                geometry_loss = torch.mean((predicted_radius - true_radius)**2)
                
                # 总损失 = 负对数似然 + 几何约束
                loss = nll_loss + 0.1 * geometry_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Round {self.rounds}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        self.rounds += 1
        self.training_history.append({
            'round': self.rounds,
            'losses': losses,
            'final_loss': losses[-1]
        })

        return {
            'round': self.rounds,
            'final_loss': losses[-1],
            'loss_history': losses
        }

    def sample_posterior(self, x: torch.Tensor, context: torch.Tensor,
                        num_samples: int = 1000, device: str = 'cpu') -> torch.Tensor:
        """
        从训练好的后验分布采样
        """
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            context = context.to(device)

            samples = self.posterior_net.sample(x, context, num_samples)
            return samples

    def save_checkpoint(self, path: str):
        """保存模型检查点"""
        checkpoint = {
            'posterior_net_state_dict': self.posterior_net.state_dict(),
            'rounds': self.rounds,
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        self.posterior_net.load_state_dict(checkpoint['posterior_net_state_dict'])
        self.rounds = checkpoint['rounds']
        self.training_history = checkpoint['training_history']


def create_sbi_model(config: dict) -> SBIModel:
    """创建SBI模型的工厂函数"""
    return SBIModel(config)


def compute_posterior_statistics(posterior_samples: torch.Tensor,
                               true_params: Optional[torch.Tensor] = None) -> dict:
    """
    计算后验分布的统计量
    Args:
        posterior_samples: 后验样本 (num_samples, B, param_dim)
        true_params: 真实参数 (B, param_dim)
    Returns:
        统计量字典
    """
    num_samples, B, param_dim = posterior_samples.shape

    # 计算均值和标准差
    mean = posterior_samples.mean(dim=0)  # (B, param_dim)
    std = posterior_samples.std(dim=0)    # (B, param_dim)

    stats = {
        'mean': mean,
        'std': std,
        'median': posterior_samples.median(dim=0).values,
        'q25': posterior_samples.quantile(0.25, dim=0),
        'q75': posterior_samples.quantile(0.75, dim=0)
    }

    # 如果提供了真实参数，计算误差统计
    if true_params is not None:
        error = mean - true_params
        abs_error = torch.abs(error)
        mse = torch.mean(error ** 2, dim=0)
        mae = torch.mean(abs_error, dim=0)

        stats.update({
            'error': error,
            'abs_error': abs_error,
            'mse': mse,
            'mae': mae
        })

    return stats
