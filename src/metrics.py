import numpy as np
import torch
from .physics import get_sigma1, get_sigma2, gen_events


@torch.no_grad()
def histogram_kl(x_obs: torch.Tensor, params: torch.Tensor, num_points: int = 10000, bins: int = 100) -> float:
    """
    用直方图KL作为观测一致性指标。
    - x_obs: 观测事件 (B, N, 2)
    - params: 参数 (B, 6)
    返回：平均KL (float)，越小越好
    """
    x_obs_np = x_obs.detach().cpu().numpy()
    params_np = params.detach().cpu().numpy()
    B = x_obs_np.shape[0]

    def _hist(x):
        h, _ = np.histogram(x, bins=bins, range=(0.1, 1.0), density=True)
        h = np.clip(h, 1e-12, None)
        h = h / np.sum(h)
        return h

    kls = []
    for i in range(B):
        # 生成模型预测下的事件分布
        s1, _, _ = gen_events(lambda _: get_sigma1(_, params_np[i]), nevents=num_points)
        s2, _, _ = gen_events(lambda _: get_sigma2(_, params_np[i]), nevents=num_points)
        h1_pred = _hist(s1)
        h2_pred = _hist(s2)

        h1_obs = _hist(x_obs_np[i, :, 0])
        h2_obs = _hist(x_obs_np[i, :, 1])

        # KL(P||Q)
        kl1 = float(np.sum(h1_obs * (np.log(h1_obs) - np.log(h1_pred))))
        kl2 = float(np.sum(h2_obs * (np.log(h2_obs) - np.log(h2_pred))))
        kls.append(0.5 * (kl1 + kl2))

    return float(np.mean(kls))


@torch.no_grad()
def norm_mse(norm_obs: torch.Tensor, params: torch.Tensor, num_points: int = 10000) -> float:
    """
    用归一化常数的一致性（MSE）作为指标。
    - norm_obs: (B, 2)
    - params: (B, 6)
    返回：平均MSE (float)，越小越好
    """
    params_np = params.detach().cpu().numpy()
    B = params_np.shape[0]
    diffs = []
    for i in range(B):
        _, n1, _ = gen_events(lambda _: get_sigma1(_, params_np[i]), nevents=num_points)
        _, n2, _ = gen_events(lambda _: get_sigma2(_, params_np[i]), nevents=num_points)
        pred = np.array([n1, n2], dtype=np.float64)
        obs = norm_obs[i].detach().cpu().numpy()
        diffs.append(np.mean((pred - obs) ** 2))
    return float(np.mean(diffs))


