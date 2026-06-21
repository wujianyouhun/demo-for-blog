"""训练引擎"""
import logging, time
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, lr=1e-4, weight_decay=1e-4, scheduler_type="cosine",
                 device="auto", mixed_precision=True):
        self.device = self._get_device(device)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=lr, weight_decay=weight_decay)
        self.scheduler = (CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)
                          if scheduler_type == "cosine"
                          else StepLR(self.optimizer, step_size=15, gamma=0.5))
        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None
        self.history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "lr": []}
        self.best_val_f1 = 0.0

    def train_epoch(self, dl):
        self.model.train()
        tl, tf, n = 0, 0, 0
        for b in dl:
            a, bi, l = b["image_a"].to(self.device), b["image_b"].to(self.device), b["label"].to(self.device)
            self.optimizer.zero_grad()
            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    out = self.model(a, bi)
                    loss = self.criterion(out["out"], l)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(a, bi)
                loss = self.criterion(out["out"], l)
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                tf += self._f1(out["out"].argmax(1), l)
            tl += loss.item()
            n += 1
        return {"loss": tl / max(n, 1), "f1": tf / max(n, 1)}

    @torch.no_grad()
    def validate(self, dl):
        self.model.eval()
        tl, tf, n = 0, 0, 0
        for b in dl:
            a, bi, l = b["image_a"].to(self.device), b["image_b"].to(self.device), b["label"].to(self.device)
            out = self.model(a, bi)
            tl += self.criterion(out["out"], l).item()
            tf += self._f1(out["out"].argmax(1), l)
            n += 1
        return {"loss": tl / max(n, 1), "f1": tf / max(n, 1)}

    def fit(self, train_loader, val_loader, epochs=50, save_dir=None, early_stopping_patience=10):
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        patience = 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tm = self.train_epoch(train_loader)
            vm = self.validate(val_loader)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            for k, v in [("train_loss", tm["loss"]), ("val_loss", vm["loss"]),
                         ("train_f1", tm["f1"]), ("val_f1", vm["f1"]), ("lr", lr)]:
                self.history[k].append(v)
            logger.info(f"Epoch [{ep}/{epochs}] loss={tm['loss']:.4f} f1={tm['f1']:.4f} | "
                        f"val_loss={vm['loss']:.4f} val_f1={vm['f1']:.4f} lr={lr:.6f} {time.time()-t0:.1f}s")
            if vm["f1"] > self.best_val_f1:
                self.best_val_f1 = vm["f1"]
                patience = 0
                if save_dir:
                    torch.save(self.model.state_dict(), Path(save_dir) / "best_model.pth")
            else:
                patience += 1
            if patience >= early_stopping_patience:
                logger.info(f"早停: {patience} epoch")
                break
        if save_dir:
            torch.save(self.model.state_dict(), Path(save_dir) / "final_model.pth")
        return self.history

    @staticmethod
    def _f1(p, l):
        tp = ((p == 1) & (l == 1)).sum().item()
        fp = ((p == 1) & (l == 0)).sum().item()
        fn = ((p == 0) & (l == 1)).sum().item()
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        return 2 * pr * rc / max(pr + rc, 1e-8)

    @staticmethod
    def _get_device(d):
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else
                                ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        return torch.device(d)
