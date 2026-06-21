"""GeoAI Core - 语义分割模型训练器"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """
    语义分割训练器, 支持混合精度训练、多种学习率调度器和早停。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[str | torch.device] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 混合精度
        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # 学习率调度器
        self.scheduler = self._build_scheduler()

        # 早停
        self.patience = config.get("early_stopping_patience", 10)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # 训练记录
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": [],
            "lr": [],
        }

        self.epochs = config.get("epochs", 50)
        self.num_classes = config.get("num_classes", 6)

        logger.info(
            "Trainer 初始化: device=%s, amp=%s, epochs=%d, patience=%d",
            self.device, self.use_amp, self.epochs, self.patience,
        )

    def _build_scheduler(self):
        sched_type = self.config.get("lr_scheduler", "cosine")
        if sched_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-7
            )
        elif sched_type == "step":
            return StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        elif sched_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            logger.warning("未知调度器 '%s', 使用 cosine", sched_type)
            return CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-7
            )

    def _train_one_epoch(self) -> float:
        """训练一个 epoch, 返回平均损失。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple:
        """验证一个 epoch, 返回 (val_loss, mean_iou)。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # IoU 计算
        intersection = np.zeros(self.num_classes, dtype=np.float64)
        union = np.zeros(self.num_classes, dtype=np.float64)

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            # 计算 IoU
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            for cls in range(self.num_classes):
                pred_mask = (preds == cls)
                target_mask = (targets == cls)
                intersection[cls] += np.logical_and(pred_mask, target_mask).sum()
                union[cls] += np.logical_or(pred_mask, target_mask).sum()

        val_loss = total_loss / max(num_batches, 1)

        # mean IoU (忽略 union=0 的类别)
        valid = union > 0
        if valid.any():
            iou_per_class = intersection[valid] / union[valid]
            mean_iou = float(iou_per_class.mean())
        else:
            mean_iou = 0.0

        return val_loss, mean_iou

    def train(self) -> dict:
        """
        运行完整训练循环。

        Returns:
            训练历史字典
        """
        logger.info("=" * 60)
        logger.info("开始训练 (共 %d epochs)", self.epochs)
        logger.info("=" * 60)

        try:
            from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
            use_rich = True
        except ImportError:
            use_rich = False

        if use_rich:
            progress = Progress(
                TextColumn("[bold blue]Epoch"),
                BarColumn(),
                TextColumn("[bold]{task.fields[status]}"),
                TimeElapsedColumn(),
            )
            task = progress.add_task("训练", total=self.epochs, status="...")
            progress.start()

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss = self._train_one_epoch()

            # 验证
            val_loss, val_miou = self._validate()

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)
            self.history["lr"].append(current_lr)

            status = (
                f"[{epoch}/{self.epochs}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"mIoU={val_miou:.4f} "
                f"lr={current_lr:.2e}"
            )

            if use_rich:
                progress.update(task, advance=1, status=status)
            else:
                logger.info(status)

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                logger.info(
                    "早停触发: %d 个 epoch 无改善 (best_val_loss=%.4f)",
                    self.patience, self.best_val_loss,
                )
                break

        elapsed = time.time() - start_time

        if use_rich:
            progress.stop()

        logger.info("=" * 60)
        logger.info("训练完成! 耗时 %.1f 秒, 最佳 val_loss=%.4f", elapsed, self.best_val_loss)
        logger.info("=" * 60)

        return self.history

    def save_checkpoint(self, path: str | Path) -> Path:
        """保存模型检查点。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info("检查点已保存: %s", path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict:
        """加载模型检查点。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"检查点不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)

        logger.info("检查点已加载: %s (best_val_loss=%.4f)", path, self.best_val_loss)
        return checkpoint
