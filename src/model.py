import math
import torch
import torch.nn as nn

def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip_proj(x)

class AccurateSciDAC(nn.Module):
    def __init__(self, d_model=128, time_embed_dim=32, use_transformer=True, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.time_embed_dim = time_embed_dim
        self.use_transformer = use_transformer

        self.pointnet = nn.Sequential(
            nn.Conv1d(2, d_model // 2, 1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1)
        )

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cond_proj = ResidualMLP(d_model*3 + 6 + 2 + time_embed_dim, d_model * 2, d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 6)
        )

    def forward(self, x, norm, noisy_params, t):
        B, N, _ = x.shape
        x = x.transpose(1, 2)
        feats = self.pointnet(x)
        max_feat = torch.max(feats, dim=2).values
        mean_feat = torch.mean(feats, dim=2)
        min_feat = torch.min(feats, dim=2).values

        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        t_embed = sinusoidal_embedding(t.float(), self.time_embed_dim)
        t_embed = t_embed.view(t_embed.shape[0], -1)

        cond = torch.cat([max_feat, mean_feat, min_feat, norm, noisy_params, t_embed], dim=1)
        cond = self.cond_proj(cond).unsqueeze(1)

        if self.use_transformer:
            cond = self.transformer(cond)

        out = self.out_proj(cond.squeeze(1))
        return out

    def reverse_sample(self, x, norm, T=100, num_samples=10, device='cpu'):
        from .diffusion import get_alpha_schedule
        beta, alpha, alpha_bar = get_alpha_schedule(T)
        samples = []
        B, N, D = x.size()
        x, norm = x.to(device), norm.to(device)
        for _ in range(num_samples):
            xt = torch.randn(B, 6, device=device)
            for t_step in reversed(range(T)):
                t = torch.full((B,), t_step, device=device, dtype=torch.long)
                with torch.no_grad():
                    noise_pred = self.forward(x, norm, xt, t)
                a_t = alpha[t_step].to(device).view(1)
                ab_t = alpha_bar[t_step].to(device).view(1)
                beta_t = beta[t_step].to(device).view(1)
                noise = torch.randn_like(xt) if t_step > 0 else 0
                xt = (1 / torch.sqrt(a_t)) * (xt - ((1 - a_t) / torch.sqrt(1 - ab_t)) * noise_pred) + torch.sqrt(beta_t) * noise
            samples.append(xt)
        return torch.stack(samples)


