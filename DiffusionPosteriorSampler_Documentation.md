# DiffusionPosteriorSampler 结构文档

## 类定义

```python
class DiffusionPosteriorSampler(torch.nn.Module):
```

## 完整代码

```python
class DiffusionPosteriorSampler(torch.nn.Module):
    def __init__(self, y_dim, x_dim, n_summaries,
                 num_hidden_layer,device,use_encoder, data_type="iid", sigma_data=0.5):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.use_encoder = use_encoder
        self.n_summaries = n_summaries if use_encoder else y_dim

        if self.use_encoder:
            if data_type == "iid":
                self.summary = DeepSetSummary(y_dim, n_summaries).to(device)
                print("Encoder is for iid data. If not, please check it.")
            elif data_type == "time":
                self.summary = BayesFlowEncoder(y_dim, n_summaries).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            elif data_type == "set":
                num_head = 4
                num_seed = 4
                self.summary = SetEmbedderClean(y_dim, n_summaries, num_head, num_seed).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            else:
                raise ImportError("Other summary is not supported")
        else:
            pass

        self.decoder = ScoreNetwork(
            x_dim=x_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=self.n_summaries,
            cond_mask_prob=0.0,
            num_hidden_layers=num_hidden_layer,
            output_dim=x_dim,
            device=device,
            cond_conditional=True).to(device)
        # self.diffusion = VariancePreservingSDE(action_dim=x_dim, state_dim=n_summaries, device=device)
        self.diffusion = KarrasSDE(theta_dim=x_dim, data_dim=self.n_summaries, device=device, sigma_data=sigma_data)

    @torch.no_grad()
    def sample(self, y, num_steps=18):
        with torch.no_grad():
            s = self.summary(y) if self.use_encoder else y
            # z, log_p,_ = self.diffusion.sample(self.decoder,s,num_steps)
            z = self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)
        return z

    @torch.no_grad()
    def sample_given_s(self,s, num_steps=18):
        z = self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)
        return z

    def loss(self, x, y):
        s = self.summary(y) if self.use_encoder else y
        # diffusion_loss = self.diffusion.diffusion_loss(self.decoder, x, s).mean()
        diffusion_loss = self.diffusion.diffusion_train_step(self.decoder, x, s)
        return diffusion_loss
```

## 类结构说明

### 初始化参数 (`__init__`)

| 参数 | 类型 | 说明 |
|------|------|------|
| `y_dim` | int | 观测数据的维度 |
| `x_dim` | int | 参数空间的维度 |
| `n_summaries` | int | 摘要统计量的数量 |
| `num_hidden_layer` | int | 解码器网络隐藏层数量 |
| `device` | torch.device | 计算设备 (CPU/GPU) |
| `use_encoder` | bool | 是否使用编码器 |
| `data_type` | str | 数据类型，可选: "iid", "time", "set" (默认: "iid") |
| `sigma_data` | float | 数据标准差参数 (默认: 0.5) |

### 类属性

#### 基本属性
- `self.y_dim`: 观测数据维度
- `self.x_dim`: 参数空间维度
- `self.use_encoder`: 是否使用编码器标志
- `self.n_summaries`: 摘要统计量维度（如果使用编码器则为 `n_summaries`，否则为 `y_dim`）

#### 编码器 (`self.summary`)
根据 `data_type` 选择不同的编码器：
- **"iid"**: 使用 `DeepSetSummary` - 适用于独立同分布数据
- **"time"**: 使用 `BayesFlowEncoder` - 适用于时间依赖数据
- **"set"**: 使用 `SetEmbedderClean` - 适用于集合数据（使用 4 个 head 和 4 个 seed）

如果 `use_encoder=False`，则不使用编码器，直接使用原始观测数据 `y`。

#### 解码器 (`self.decoder`)
使用 `ScoreNetwork` 作为解码器，配置参数：
- `x_dim`: 参数空间维度
- `hidden_dim`: 隐藏层维度 (固定为 256)
- `time_embed_dim`: 时间嵌入维度 (固定为 16)
- `cond_dim`: 条件维度 (等于 `self.n_summaries`)
- `cond_mask_prob`: 条件掩码概率 (固定为 0.0)
- `num_hidden_layers`: 隐藏层数量 (由 `num_hidden_layer` 参数指定)
- `output_dim`: 输出维度 (等于 `x_dim`)
- `device`: 计算设备
- `cond_conditional`: 条件标志 (固定为 True)

#### 扩散过程 (`self.diffusion`)
使用 `KarrasSDE` 作为扩散过程：
- `theta_dim`: 参数维度 (等于 `x_dim`)
- `data_dim`: 数据维度 (等于 `self.n_summaries`)
- `device`: 计算设备
- `sigma_data`: 数据标准差参数

**注意**: 代码中注释掉了 `VariancePreservingSDE` 的选项，当前使用 `KarrasSDE`。

### 方法说明

#### 1. `sample(y, num_steps=18)`
**功能**: 从观测数据 `y` 生成样本

**参数**:
- `y`: 观测数据
- `num_steps`: 采样步数 (默认: 18)

**流程**:
1. 如果使用编码器，通过 `self.summary(y)` 获取摘要统计量 `s`；否则直接使用 `y` 作为 `s`
2. 调用 `self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)` 进行采样
3. 返回采样结果 `z`

**装饰器**: `@torch.no_grad()` - 禁用梯度计算

#### 2. `sample_given_s(s, num_steps=18)`
**功能**: 给定摘要统计量 `s` 直接生成样本

**参数**:
- `s`: 摘要统计量
- `num_steps`: 采样步数 (默认: 18)

**流程**:
1. 直接调用 `self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)` 进行采样
2. 返回采样结果 `z`

**装饰器**: `@torch.no_grad()` - 禁用梯度计算

**区别**: 此方法跳过了编码步骤，直接使用提供的摘要统计量。

#### 3. `loss(x, y)`
**功能**: 计算训练损失

**参数**:
- `x`: 真实参数样本
- `y`: 观测数据

**流程**:
1. 如果使用编码器，通过 `self.summary(y)` 获取摘要统计量 `s`；否则直接使用 `y` 作为 `s`
2. 调用 `self.diffusion.diffusion_train_step(self.decoder, x, s)` 计算扩散训练损失
3. 返回损失值

**注意**: 代码中注释掉了 `diffusion_loss` 的选项，当前使用 `diffusion_train_step`。

## 依赖模块

### 导入的类
```python
from .summary import DeepSetSummary, BayesFlowEncoder, SetEmbedderClean
from .diffusion import KarrasSDE
from .utils import ScoreNetwork
```

### 主要组件
1. **编码器**: `DeepSetSummary`, `BayesFlowEncoder`, `SetEmbedderClean` - 用于将观测数据编码为摘要统计量
2. **解码器**: `ScoreNetwork` - 用于学习分数函数（score function）
3. **扩散过程**: `KarrasSDE` - 用于实现扩散模型的采样过程

## 使用示例

```python
# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionPosteriorSampler(
    y_dim=10,              # 观测数据维度
    x_dim=2,               # 参数空间维度
    n_summaries=4,         # 摘要统计量数量
    num_hidden_layer=4,    # 隐藏层数量
    device=device,         # 计算设备
    use_encoder=True,      # 使用编码器
    data_type="iid",       # 数据类型
    sigma_data=0.5         # 数据标准差
)

# 训练
x = ...  # 真实参数样本
y = ...  # 观测数据
loss = model.loss(x, y)
loss.backward()

# 采样
y_obs = ...  # 观测数据
samples = model.sample(y_obs, num_steps=18)

# 给定摘要统计量采样
s = ...  # 摘要统计量
samples = model.sample_given_s(s, num_steps=18)
```

## 数据流图

```
观测数据 y
    ↓
[可选: 编码器 summary] → 摘要统计量 s
    ↓
ScoreNetwork (decoder) ← 条件信息 s
    ↓
KarrasSDE (diffusion) → 采样 z
```

## 注意事项

1. **编码器选择**: 根据数据类型 (`data_type`) 选择合适的编码器，不匹配的类型可能导致性能下降
2. **设备管理**: 所有组件都会移动到指定的 `device` 上
3. **梯度计算**: `sample` 和 `sample_given_s` 方法使用 `@torch.no_grad()` 装饰器，不会计算梯度
4. **采样步数**: `num_steps` 参数控制采样质量，步数越多质量越好但计算时间越长

