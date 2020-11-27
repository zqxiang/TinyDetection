from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List

import torch.nn as nn

from tinydet.layers import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self._out_features: List[str] = []
        self._out_feature_channels: Dict[str, int] = {}
        self._out_feature_strides: Dict[str, int] = {}

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self):
        return 0

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            ) for name in self._out_features
        }
