# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

# from .metrics import bbox_iou, probiou
# from .tal import bbox2dist

## Ultralytics 官方的 VarifocalLoss
class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()
    

## 改写的VarifocalLoss
class VarifocalLoss_new(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, _fg_mask, alpha=0.75, gamma=2.0):
        """Computes varfocal loss.
        Args:
            pred_score (Tensor): The predicted score(b, achor_number, nc).
            gt_score (Tensor): The ground truth score(b, achor_number, nc).
            fg_mask (Tensor): The foreground mask((b, achor_number).
            alpha (float): The alpha parameter.
            gamma (float): The gamma parameter.
        """
        fg_mask = _fg_mask.float().unsqueeze(-1).expand_as(pred_score)
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - fg_mask) + gt_score * fg_mask
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), (gt_score*fg_mask).float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss
    
if __name__ == '__main__':
    pred_score = torch.randn(4, 8400, 4)
    gt_score = torch.randn(4, 8400, 4)
    fg_mask = torch.randint(0, 2, (4, 8400), dtype=torch.bool)
    
    # VarifocalLoss_new
    loss = VarifocalLoss_new()
    
    vfl = loss(pred_score, gt_score, fg_mask)