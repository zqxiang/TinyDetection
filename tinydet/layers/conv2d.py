import torch.nn as nn
from torch import Tensor

from .activation import get_activation
from .batch_norm import get_norm

__all__ = ["Conv2d"]


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        norm: str = "",
        activation: str = "",
        **kwargs
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.norm = get_norm(norm, out_channels)

        inplace = kwargs.get("inplace", False)
        self.activation = get_activation(activation, inplace)

    def forward(self, x: Tensor):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
