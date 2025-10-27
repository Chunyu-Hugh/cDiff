import numpy as np
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


