import numpy as np
import torch
from .physics import (
    pmin, pmax, amin, amax, bmin, bmax, qmin, qmax, cmin, cmax, dmin, dmax,
    pmin_r, pmax_r, amin_r, amax_r, bmin_r, bmax_r, qmin_r, qmax_r, cmin_r, cmax_r, dmin_r, dmax_r,
    get_sigma1, get_sigma2, gen_events
)

def generate_data_batch(batch_size, num_points):
    p = np.random.uniform(pmin, pmax, (batch_size,1))
    a = np.random.uniform(amin, amax, (batch_size,1))
    b = np.random.uniform(bmin, bmax, (batch_size,1))
    q = np.random.uniform(qmin, qmax, (batch_size,1))
    c = np.random.uniform(cmin, cmax, (batch_size,1))
    d = np.random.uniform(dmin, dmax, (batch_size,1))

    param = np.concatenate((p,a,b,q,c,d), axis=1)

    x_list, norm_list = [], []
    for i in range(batch_size):
        s1_events, s1_norm, _ = gen_events(lambda _: get_sigma1(_, param[i]), nevents=num_points)
        s2_events, s2_norm, _ = gen_events(lambda _: get_sigma2(_, param[i]), nevents=num_points)
        events = np.stack([s1_events, s2_events], axis=1)
        x_list.append(events)
        norm_list.append([s1_norm, s2_norm])

    x = torch.from_numpy(np.stack(x_list)).float()       # (B, N, 2)
    norm = torch.from_numpy(np.stack(norm_list)).float() # (B, 2)
    param = torch.from_numpy(param).float()              # (B, 6)
    return x, norm, param

def clamp_to_inference_bounds(params: torch.Tensor):
    lower = torch.tensor([pmin_r, amin_r, bmin_r, qmin_r, cmin_r, dmin_r], device=params.device)
    upper = torch.tensor([pmax_r, amax_r, bmax_r, qmax_r, cmax_r, dmax_r], device=params.device)
    return torch.max(torch.min(params, upper), lower)


