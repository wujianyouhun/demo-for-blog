from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion

from .config import IGNORE_INDEX


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    pred = np.asarray(prediction).reshape(-1)
    truth = np.asarray(target).reshape(-1)
    valid = (truth != ignore_index) & (truth >= 0) & (truth < num_classes) & (pred >= 0) & (pred < num_classes)
    encoded = truth[valid] * num_classes + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    matrix = matrix.astype(np.float64)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = support + predicted - tp
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    dice = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    valid = support > 0
    return {
        "pixel_accuracy": float(tp.sum() / max(matrix.sum(), 1)),
        "miou": float(iou[valid].mean()) if valid.any() else 0.0,
        "mdice": float(dice[valid].mean()) if valid.any() else 0.0,
        "per_class": [
            {"precision": float(precision[i]), "recall": float(recall[i]), "iou": float(iou[i]), "dice": float(dice[i]), "support": int(support[i])}
            for i in range(len(tp))
        ],
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def boundary_f1(prediction: np.ndarray, target: np.ndarray, num_classes: int, tolerance: int = 2) -> float:
    scores = []
    structure = np.ones((3, 3), dtype=bool)
    for class_id in range(num_classes):
        pred = prediction == class_id
        truth = target == class_id
        if not truth.any():
            continue
        pred_boundary = pred ^ binary_erosion(pred, structure=structure, border_value=0)
        truth_boundary = truth ^ binary_erosion(truth, structure=structure, border_value=0)
        pred_dilated = binary_dilation(pred_boundary, iterations=tolerance)
        truth_dilated = binary_dilation(truth_boundary, iterations=tolerance)
        precision = (pred_boundary & truth_dilated).sum() / max(pred_boundary.sum(), 1)
        recall = (truth_boundary & pred_dilated).sum() / max(truth_boundary.sum(), 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-8))
    return float(np.mean(scores)) if scores else 0.0


class CrossEntropyDiceLoss(torch.nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX, ce_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1)
        probabilities = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice_loss = 1 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
        return self.ce_weight * ce + (1 - self.ce_weight) * dice_loss
