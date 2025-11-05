import numpy as np
import sys

from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader

from scipy.integrate import quad
from scipy import interpolate

# Physical parameter ranges (training)
pmin, pmax = 0.0, 1.0
amin, amax = -1.0, 0.0
bmin, bmax = 0.0, 1.0
qmin, qmax = 0.0, 1.0
cmin, cmax = 0.0, 1.0
dmin, dmax = 0.0, 1.0

# Inference-time broader ranges
pmin_r, pmax_r = -10.0, 10.0
amin_r, amax_r = -10.0, 10.0
bmin_r, bmax_r = -10.0, 10.0
qmin_r, qmax_r = -10.0, 10.0
cmin_r, cmax_r = -10.0, 10.0
dmin_r, dmax_r = -10.0, 10.0

get_u = lambda x, a, b, p: p * x ** a * (1 - x) ** b
get_d = lambda x, a, b, q: q * x ** a * (1 - x) ** b

def get_sigma1(x, p):
    u = get_u(x, p[1], p[2], p[0])
    d = get_d(x, p[4], p[5], p[3])
    return 4 * u + d

def get_sigma2(x, p):
    u = get_u(x, p[1], p[2], p[0])
    d = get_d(x, p[4], p[5], p[3])
    return 4 * d + u

def gen_events(sigma, nevents, xmin=0.1, xmax=1.0):
    norm = quad(sigma, xmin, xmax)[0]
    pdf = lambda x: sigma(x) / norm
    get_cdf = lambda x: quad(pdf, x, xmax)[0]
    xs = np.linspace(xmin, xmax)
    invcdf = interpolate.interp1d([get_cdf(_) for _ in xs], xs, bounds_error=False, fill_value=0)
    u = np.random.uniform(0, 1, nevents)
    events = invcdf(u)
    return events, norm, pdf


def sample_theta():
    p = np.random.uniform(pmin, pmax)
    a = np.random.uniform(amin, amax)
    b = np.random.uniform(bmin, bmax)
    q = np.random.uniform(qmin, qmax)
    c = np.random.uniform(cmin, cmax)
    d = np.random.uniform(dmin, dmax)
    return {"p": p, "a": a, "b": b, "q": q, "c": c, "d": d}


def sample_physics_data(parms_dict, sample_size):
    theta = np.array([
        parms_dict["p"], parms_dict["a"], parms_dict["b"],
        parms_dict["q"], parms_dict["c"], parms_dict["d"]
    ])
    s1_events, _, _ = gen_events(lambda x: get_sigma1(x, theta), nevents=sample_size)
    s2_events, _, _ = gen_events(lambda x: get_sigma2(x, theta), nevents=sample_size)
    return np.stack([s1_events, s2_events], axis=1)


def return_physics_dl(n_batches=256, batch_size=128, n_sample=None, return_ds=False):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample + 1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=100, high=1000):
            return np.random.randint(low=low, high=high, size=n)

    ds = BayesDataStream(
        n_batches=n_batches,
        batch_size=batch_size,
        sample_theta=sample_theta,
        sample_y=sample_physics_data,
        sample_n=my_gen_sample_size,
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl, ds
    else:
        return dl


