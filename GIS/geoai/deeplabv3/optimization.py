"""遥感分割的损失、采样、增强和统一评价；不改变原始数据。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from common import IGNORE


def paired_transform(image, target, flip_x=False, flip_y=False, rotations=0):
    if flip_x:
        image, target = image.flip(-1), target.flip(-1)
    if flip_y:
        image, target = image.flip(-2), target.flip(-2)
    return (torch.rot90(image, rotations, (-2, -1)).contiguous(),
            torch.rot90(target, rotations, (-2, -1)).contiguous())


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augment=True):
        self.dataset, self.augment = dataset, augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        if self.augment:
            image, target = paired_transform(image, target, bool(torch.rand(()) < .5),
                                             bool(torch.rand(()) < .5), int(torch.randint(4, ())))
        return image, target


def dataset_counts(dataset):
    rows = []
    for _, target in dataset:
        rows.append(torch.bincount(target[target != IGNORE], minlength=dataset.num_classes))
    return torch.stack(rows)


def sampling_weights(counts, max_ratio=3.):
    """少数类占比提升切片概率，最大/最小抽样权重比受限。"""
    if max_ratio < 1 or not np.isfinite(max_ratio):
        raise ValueError("max_ratio 必须为不小于 1 的有限数。")
    counts = counts.double()
    totals = counts.sum(0)
    if (totals <= 0).any() or (counts.sum(1) <= 0).any():
        raise ValueError("训练类或切片缺少有效像元。")
    fractions = counts / counts.sum(1, keepdim=True)
    score = (fractions * totals.rsqrt()).sum(1)
    return (score / score.min()).clamp(max=max_ratio)


def segmentation_loss(logits, target, class_weights=None, dice_weight=.3):
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0
    ce = nn.functional.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE)
    if not dice_weight:
        return ce
    probs = logits.softmax(1) * valid[:, None]
    safe_target = target.masked_fill(~valid, 0)
    truth = nn.functional.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2)
    truth = truth * valid[:, None]
    present = truth.sum((0, 2, 3)) > 0
    intersection = (probs * truth).sum((0, 2, 3))
    denominator = probs.sum((0, 2, 3)) + truth.sum((0, 2, 3))
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return ce + dice_weight * (1 - dice[present]).mean()


def add_confusion(matrix, prediction, target):
    valid = target != IGNORE
    n = matrix.shape[0]
    matrix += torch.bincount(target[valid] * n + prediction[valid], minlength=n*n).reshape(n, n)


def metrics_from_confusion(confusion, names):
    m = torch.as_tensor(confusion, dtype=torch.float64, device="cpu")
    support, predicted, tp = m.sum(1), m.sum(0), m.diag()
    union = support + predicted - tp
    ratio = lambda a, b: float(a / b) if b else None
    per_class = [dict(class_id=i, name=name, pixels=int(support[i]), predicted_pixels=int(predicted[i]),
                      iou=ratio(tp[i], union[i]), precision=ratio(tp[i], predicted[i]),
                      recall=ratio(tp[i], support[i]), f1=ratio(2 * tp[i], support[i] + predicted[i]))
                 for i, name in enumerate(names)]
    ious = [x["iou"] for x in per_class if x["iou"] is not None]
    return dict(confusion_matrix=m.long().tolist(), per_class=per_class,
                per_class_iou=[x["iou"] for x in per_class],
                miou=sum(ious) / len(ious) if ious else None,
                accuracy=ratio(tp.sum(), support.sum()), valid_pixels=int(support.sum()),
                zero_prediction_classes=[x["name"] for x in per_class if x["pixels"] and not x["predicted_pixels"]])


def epoch_pass(model, loader, device, names, optimizer=None, class_weights=None,
               dice_weight=0., grad_clip=1., freeze_backbone=False):
    training = optimizer is not None
    model.train(training)
    if training:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        if freeze_backbone:
            model.backbone.eval()
    matrix = torch.zeros((len(names), len(names)), dtype=torch.int64, device=device)
    ce_sum, objective_sum, total = 0., 0., 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            count = int((targets != IGNORE).sum())
            if not count:
                continue
            logits = model(images)["out"]
            loss = segmentation_loss(logits, targets, class_weights, dice_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("损失出现 NaN/Inf，请检查输入与学习率。")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            ce = nn.functional.cross_entropy(logits.detach(), targets, ignore_index=IGNORE, reduction="sum")
            ce_sum += float(ce)
            # 组合目标按批次有效像元加权记录；它不是全数据集 Dice。
            objective_sum += float(loss.detach()) * count
            total += count
            add_confusion(matrix, logits.detach().argmax(1), targets)
    if not total:
        raise ValueError("数据集没有有效标签像元。")
    result = metrics_from_confusion(matrix.cpu(), names)
    result.update(ce_loss=ce_sum / total, objective=objective_sum / total)
    return result
