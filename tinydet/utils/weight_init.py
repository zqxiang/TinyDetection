import torch.nn as nn

from tinydet.layers import Conv2d

__all__ = ["msra_fill", "xavier_fill", "normal_fill"]


def init_deco(func):
    def wrapper(module):
        if isinstance(module, Conv2d):
            func(module.conv)
        else:
            func(module)

    return wrapper


@init_deco
def msra_fill(module):
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_deco
def xavier_fill(module):
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias)


@init_deco
def normal_fill(module):
    nn.init.normal_(module.weight, mean=0, std=0.01)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
