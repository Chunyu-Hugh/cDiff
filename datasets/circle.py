import numpy as np
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader

# Hyperparameters for the circle dataset
# New convention (per user request):
#   - theta = coordinate point on circle: (x, y)
#   - y     = observed radius z (optionally noisy), provided as a set of repeated scalars of length sample_size
hyperpar_dict = {
    "z_min": 0.0,
    "z_max": 2.0,
    # Standard deviation of Gaussian noise added to observed radius y
    # Set via set_circle_noise_sigma(sigma)
    "noise_sigma": 0.0,
}


def set_circle_noise_sigma(sigma: float):
    """
    Set the Gaussian noise std added to observed radius values.
    """
    hyperpar_dict["noise_sigma"] = float(sigma)


def sample_theta():
    """
    Sample a coordinate theta = (x, y) uniformly by first drawing a radius z ~ Uniform[z_min, z_max]
    and an angle t ~ Uniform[0, 2π], then mapping to (x, y) = (z cos t, z sin t).

    Returns a dict for compatibility with BayesDataStream, which will be flattened to [x, y].
    """
    z = np.random.uniform(hyperpar_dict["z_min"], hyperpar_dict["z_max"])
    t = np.random.uniform(0.0, 2.0 * np.pi)
    x = z * np.cos(t)
    y = z * np.sin(t)
    return {"x": x, "y": y}


def sample_circle_radius(parms_dict, sample_size):
    """
    Given theta = (x, y), produce observed radius values y_obs.
    We return a set of repeated scalar observations of length `sample_size` (shape: [sample_size, 1])
    so it is compatible with the existing set-encoder interface.
    """
    x = float(parms_dict["x"])  # coordinate
    y = float(parms_dict["y"])  # coordinate
    z = np.sqrt(x * x + y * y)

    sigma = hyperpar_dict.get("noise_sigma", 0.0)
    if sigma and sigma > 0.0:
        obs = z + np.random.normal(0.0, sigma, size=(sample_size,))
    else:
        obs = np.full((sample_size,), z)
    return obs.reshape(sample_size, 1)


def sample_points_given_z(z_value, sample_size):
    """
    Sample (x, y) given a fixed z (radius). Useful for conditional training.
    Returns ndarray of shape (sample_size, 2).
    """
    z = float(z_value)
    t = np.random.uniform(0.0, 2.0 * np.pi, size=sample_size)
    x = z * np.cos(t)
    y = z * np.sin(t)
    samples = np.stack([x, y], axis=1)

    sigma = hyperpar_dict.get("noise_sigma", 0.0)
    if sigma and sigma > 0.0:
        samples = samples + np.random.normal(0.0, sigma, size=samples.shape)

    return samples


def return_circle_dl(n_batches=256, batch_size=128, n_sample=None, return_ds=False):
    if n_sample is not None:
        def fixed_sizes(n, low=n_sample, high=n_sample + 1):
            return np.random.randint(low=low, high=high, size=n)
        sample_n = fixed_sizes
    else:
        def default_sizes(n, low=100, high=1000):
            return np.random.randint(low=low, high=high, size=n)
        sample_n = default_sizes

    ds = BayesDataStream(
        n_batches=n_batches,
        batch_size=batch_size,
        sample_theta=sample_theta,
        sample_y=sample_circle_radius,
        sample_n=sample_n,
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl, ds
    else:
        return dl


if __name__ == "__main__":
    dl = return_circle_dl(n_batches=2, batch_size=2, n_sample=5)
    for theta, y in dl:
        print(theta.shape)  # (B, 2) now theta=(x, y)
        print(y.shape)      # (B, n_sample, 1) now y=radius set

