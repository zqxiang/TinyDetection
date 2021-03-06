import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "FrozenBatchNorm2d",
    "get_norm",
]


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x: Tensor):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

    def __repr__(self) -> str:
        return f"FrozenBatchNorm2d(num_feature={self.num_features}, eps={self.eps})"

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def get_norm(norm: str, out_channels: int):
    if len(norm) == 0:
        return None

    norm = {
        "BN": nn.BatchNorm2d,
        "GN": lambda x: nn.GroupNorm(32, x),
        "FrozenBN": FrozenBatchNorm2d,
        "SyncBN": nn.SyncBatchNorm,
    }[norm]
    return norm(out_channels)
