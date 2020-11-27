import torch
from torch import Tensor

__all__ = ["smooth_l1_loss"]


def smooth_l1_loss(inputs: Tensor, targets: Tensor, beta: float, reduction: str = "none") -> Tensor:
    if beta < 1e-5:
        loss = torch.abs(inputs - targets)
    else:
        n = torch.abs(inputs - targets)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
