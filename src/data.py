import numpy as np
import torch
from .physics import (
    pmin, pmax, amin, amax, bmin, bmax, qmin, qmax, cmin, cmax, dmin, dmax,
    pmin_r, pmax_r, amin_r, amax_r, bmin_r, bmax_r, qmin_r, qmax_r, cmin_r, cmax_r, dmin_r, dmax_r,
    get_sigma1, get_sigma2, gen_events
)
from datasets.BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader

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



# ========== Witch-hat style streaming API for physics data (non-breaking) ==========
def _sample_theta_physics():
    """
    Sample parameters using the same training-time ranges as generate_data_batch.
    Returns a dict so it is compatible with BayesDataStream.
    """
    p = np.random.uniform(pmin, pmax)
    a = np.random.uniform(amin, amax)
    b = np.random.uniform(bmin, bmax)
    q = np.random.uniform(qmin, qmax)
    c = np.random.uniform(cmin, cmax)
    d = np.random.uniform(dmin, dmax)
    return {"p": p, "a": a, "b": b, "q": q, "c": c, "d": d}


def _sample_physics_events(parms_dict, sample_size):
    """
    Generate event pairs (sigma1, sigma2) using existing physics logic.
    - parms_dict: dict with keys p,a,b,q,c,d
    - sample_size: number of events per sigma
    Returns np.ndarray of shape (sample_size, 2)
    """
    theta = np.array([
        parms_dict["p"], parms_dict["a"], parms_dict["b"],
        parms_dict["q"], parms_dict["c"], parms_dict["d"]
    ])
    s1_events, _, _ = gen_events(lambda x: get_sigma1(x, theta), nevents=sample_size)
    s2_events, _, _ = gen_events(lambda x: get_sigma2(x, theta), nevents=sample_size)
    return np.stack([s1_events, s2_events], axis=1)


def _default_sample_sizes(n, low=100, high=1000):
    return np.random.randint(low=low, high=high, size=n)


def return_physics_dl(n_batches=256, batch_size=128, n_sample=None, return_ds=False):
    """
    Witch-hat style DataLoader for physics simulator.
    - Preserves original generation logic (same parameter ranges and event mechanism).
    - Yields (theta, y) where theta is (6,) and y is (sample_size, 2).
    """
    if n_sample is not None:
        def fixed_sizes(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
        sample_n = fixed_sizes
    else:
        sample_n = _default_sample_sizes

    ds = BayesDataStream(
        n_batches=n_batches,
        batch_size=batch_size,
        sample_theta=_sample_theta_physics,
        sample_y=_sample_physics_events,
        sample_n=sample_n,
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl, ds
    return dl

