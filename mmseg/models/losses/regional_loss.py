import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .cross_entropy_loss import _expand_onehot_labels

_EPS = 1e-10


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """
    Segmentation dice loss in http://arxiv.org/abs/1912.11619 combined with sigmoid function for multi-label setting
    The loss can be described as:
        l = 1 - (2 * intersect) / (|X^2| + |Y^2|)
    Args:
        Input : Tensor of shape (minibatch, C, h, w)
        Target : Tensor of shape (minibatch, C, h, w)
    """
    def __init__(self, log_loss=False) -> None:
        super().__init__()
        self.log_loss = log_loss

    def forward(self, pred : torch.Tensor,
                label : torch.Tensor,
                weight : torch.Tensor = None,
                ignore_index=-100) -> torch.Tensor:
        if pred.dim() != label.dim():
            assert (
                pred.dim() == 4 and label.dim() == 3
            ), (
                'Only pred shape [N, C, H, W], label shape [N, H, W] are supported'
            )
            label, weight = _expand_onehot_labels(
                label, weight, pred.shape, ignore_index)

        prob = F.softmax(pred, dim=1)
        label = label.float()
        inter = torch.einsum("bchw,bchw->bc", prob, label)
        union = torch.einsum("bchw->bc", prob) + torch.einsum("bchw->bc", label)
        dice_score = (2 * inter + _EPS) / (union + _EPS)
        if self.log_loss:
            loss = - torch.log(dice_score).mean()
        else:
            loss = (1 - dice_score).mean()

        return loss


@LOSSES.register_module()
class CrossEntropyWithSizeL1Loss(nn.Module):
    def __init__(self, alpha : float = 0.1,
                 temp : float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temp = temp

    def forward(self, pred : torch.Tensor,
                label : torch.Tensor,
                ignore_index=-100) -> torch.Tensor:
        pass