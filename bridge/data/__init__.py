import torch
from .cacheloader import CacheLoader


def logit_transform(image: torch.Tensor, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(d_config, X):
    if d_config.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    elif d_config.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if d_config.rescaled:
        X = 2 * X - 1.
    elif d_config.logit_transform:
        X = logit_transform(X)

    return X


def inverse_data_transform(d_config, X):

    if d_config.logit_transform:
        X = torch.sigmoid(X)
    elif d_config.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)