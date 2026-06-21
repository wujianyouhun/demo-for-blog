"""变化检测评价指标"""
import numpy as np


class ChangeMetrics:
    @staticmethod
    def compute(pred, gt):
        pred, gt = pred.astype(bool), gt.astype(bool)
        tp = int(np.sum(pred & gt))
        fp = int(np.sum(pred & ~gt))
        fn = int(np.sum(~pred & gt))
        tn = int(np.sum(~pred & ~gt))
        total = tp + fp + fn + tn
        oa = (tp + tn) / max(total, 1)
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-8)
        iou = tp / max(tp + fp + fn, 1)
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(total ** 2, 1)
        kappa = (oa - pe) / max(1 - pe, 1e-8)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "oa": round(oa, 4), "precision": round(pr, 4), "recall": round(rc, 4),
                "f1": round(f1, 4), "iou": round(iou, 4), "kappa": round(kappa, 4),
                "change_ratio": round(np.sum(gt) / max(total, 1), 4),
                "pred_change_ratio": round(np.sum(pred) / max(total, 1), 4)}

    @staticmethod
    def confusion_matrix(pred, gt):
        pred, gt = pred.astype(bool), gt.astype(bool)
        return np.array([[int(np.sum(~pred & ~gt)), int(np.sum(pred & ~gt))],
                          [int(np.sum(~pred & gt)), int(np.sum(pred & gt))]])

    @staticmethod
    def print_report(m):
        print("=" * 45)
        print("  变化检测评价报告")
        print("=" * 45)
        for k in ["oa", "precision", "recall", "f1", "iou", "kappa"]:
            print(f"  {k:12s}: {m[k]:.4f}")
        print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
        print("=" * 45)
