from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from .batch_norm import FrozenBatchNorm2d

__all__ = ["BaseBlock"]


class BaseBlock(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self
