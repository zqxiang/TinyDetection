from typing import List

import torch
from torch import Tensor

__all__ = ["cat", "nonzero_tuple"]


def cat(tensors: List[Tensor], dim: int = 0):
    assert isinstance(tensors, (list, tuple))

    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def nonzero_tuple(x: Tensor):
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)
