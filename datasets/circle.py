import numpy as np
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader

# Hyperparameters for the circle dataset
# x^2 + y^2 = z^2 where z is the radius (parameter to infer)
hyperpar_dict = {
    "z_min": 0.0,
    "z_max": 2.0,
    "noise_sigma": 0.0,  # add Gaussian noise to (x, y) if desired
}


def sample_theta():
    """
    Sample circle parameter z (radius) from BoxUniform over [z_min, z_max].
    For scalar z, BoxUniform reduces to Uniform(z_min, z_max).
    Returns a dict for compatibility with BayesDataStream.
    """
    z = np.random.uniform(hyperpar_dict["z_min"], hyperpar_dict["z_max"])
    return {"z": z}


def sample_circle_data(parms_dict, sample_size):
    """
    Generate (x, y) pairs on a circle of radius z.
    x = z cos(t), y = z sin(t), t ~ Uniform(0, 2π).
    Optionally adds Gaussian noise with std = noise_sigma.
    Returns ndarray of shape (sample_size, 2).
    """
    z = float(parms_dict["z"])  # radius
    t = np.random.uniform(0.0, 2.0 * np.pi, size=sample_size)
    x = z * np.cos(t)
    y = z * np.sin(t)
    samples = np.stack([x, y], axis=1)

    sigma = hyperpar_dict.get("noise_sigma", 0.0)
    if sigma and sigma > 0.0:
        samples = samples + np.random.normal(0.0, sigma, size=samples.shape)

    return samples


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
        sample_y=sample_circle_data,
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
        print(theta.shape)  # (B, 1)
        print(y.shape)      # (B, n_sample, 2)

