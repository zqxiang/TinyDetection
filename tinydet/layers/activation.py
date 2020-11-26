import torch.nn as nn


__all__ = ["get_activation"]


def get_activation(activation: str, inplace: bool = False):
    if len(activation) == 0:
        return None

    activation = {
        "ReLU": nn.ReLU,
        "ReLU6": nn.ReLU6,
        "LeakyReLU": lambda inplace: nn.LeakyReLU(0.1, inplace),
    }[activation]
    return activation(inplace)
