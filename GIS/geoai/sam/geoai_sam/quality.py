"""
标签质量评估模块

提供 IoU、Dice、面积误差、边界误差等评估指标，
用于检查 SAM 生成的标签质量。
"""

import numpy as np
from typing import Optional, Dict, Tuple, List


class QualityMetrics:
    """
    标签质量评估工具类。

    支持的指标:
        - IoU (Intersection over Union): 交并比
        - Dice Coefficient: Dice 系数
        - Precision: 精确率
        - Recall: 召回率
        - Area Error: 面积误差
        - Boundary Error: 边界误差 (Hausdorff Distance)

    典型用法:
        >>> metrics = QualityMetrics()
        >>> result = metrics.evaluate(prediction_mask, ground_truth_mask)
        >>> print(result)
    """

    @staticmethod
    def iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算 IoU (Intersection over Union)。

        IoU = |A ∩ B| / |A ∪ B|

        这是目标检测和分割中最常用的评估指标。
        - 1.0: 完美匹配
        - 0.0: 完全不匹配

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            IoU 值 (0~1)
        """
        pred = prediction.astype(bool)
        gt = ground_truth.astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return float(intersection / union)

    @staticmethod
    def dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算 Dice 系数。

        Dice = 2|A ∩ B| / (|A| + |B|)

        在医学影像和遥感领域常用。与 IoU 相比，Dice 对小目标更友好。

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            Dice 系数 (0~1)
        """
        pred = prediction.astype(bool)
        gt = ground_truth.astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        total = pred.sum() + gt.sum()

        if total == 0:
            return 1.0

        return float(2.0 * intersection / total)

    @staticmethod
    def precision(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算精确率。

        Precision = TP / (TP + FP)

        衡量预测为正的样本中有多少是真正的正样本。

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            精确率 (0~1)
        """
        pred = prediction.astype(bool)
        gt = ground_truth.astype(bool)

        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()

        if tp + fp == 0:
            return 1.0

        return float(tp / (tp + fp))

    @staticmethod
    def recall(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算召回率。

        Recall = TP / (TP + FN)

        衡量真正的正样本中有多少被预测为正。

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            召回率 (0~1)
        """
        pred = prediction.astype(bool)
        gt = ground_truth.astype(bool)

        tp = np.logical_and(pred, gt).sum()
        fn = np.logical_and(~pred, gt).sum()

        if tp + fn == 0:
            return 1.0

        return float(tp / (tp + fn))

    @staticmethod
    def f1_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算 F1 分数。

        F1 = 2 * Precision * Recall / (Precision + Recall)

        精确率和召回率的调和平均值。

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            F1 分数 (0~1)
        """
        p = QualityMetrics.precision(prediction, ground_truth)
        r = QualityMetrics.recall(prediction, ground_truth)

        if p + r == 0:
            return 0.0

        return float(2 * p * r / (p + r))

    @staticmethod
    def area_error(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        计算面积误差。

        包括绝对误差和相对误差。

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            包含绝对误差和相对误差的字典
        """
        pred = prediction.astype(bool)
        gt = ground_truth.astype(bool)

        area_pred = float(pred.sum())
        area_gt = float(gt.sum())

        abs_error = abs(area_pred - area_gt)
        rel_error = abs_error / area_gt if area_gt > 0 else float("inf")

        return {
            "area_prediction": area_pred,
            "area_ground_truth": area_gt,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "relative_error_percent": rel_error * 100,
        }

    @staticmethod
    def boundary_error(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算边界误差 (Hausdorff Distance)。

        用于检查建筑边缘、道路边缘等线性目标的精度。

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            包含边界误差指标的字典
        """
        try:
            from scipy.ndimage import distance_transform_edt

            pred = prediction.astype(bool)
            gt = ground_truth.astype(bool)

            # 提取边界
            from scipy.ndimage import binary_erosion
            pred_boundary = pred ^ binary_erosion(pred)
            gt_boundary = gt ^ binary_erosion(gt)

            # 计算距离变换
            if gt_boundary.any():
                gt_dist = distance_transform_edt(~gt_boundary)
                pred_to_gt = gt_dist[pred_boundary].mean() if pred_boundary.any() else float("inf")
            else:
                pred_to_gt = float("inf")

            if pred_boundary.any():
                pred_dist = distance_transform_edt(~pred_boundary)
                gt_to_pred = pred_dist[gt_boundary].mean() if gt_boundary.any() else float("inf")
            else:
                gt_to_pred = float("inf")

            hausdorff = max(pred_to_gt, gt_to_pred)
            avg_boundary_error = (pred_to_gt + gt_to_pred) / 2

            return {
                "hausdorff_distance": float(hausdorff),
                "pred_to_gt_mean": float(pred_to_gt),
                "gt_to_pred_mean": float(gt_to_pred),
                "avg_boundary_error": float(avg_boundary_error),
            }

        except ImportError:
            print("[质量评估] scipy 未安装，无法计算边界误差")
            return {}

    @staticmethod
    def pixel_accuracy(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算像素精度。

        Accuracy = (TP + TN) / Total

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask

        Returns:
            像素精度 (0~1)
        """
        pred = prediction.astype(bool)
        gt = ground_truth.astype(bool)

        correct = np.logical_and(pred, gt).sum() + np.logical_and(~pred, ~gt).sum()
        total = pred.size

        return float(correct / total)

    @classmethod
    def evaluate(
        cls,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        include_boundary: bool = True,
    ) -> Dict[str, float]:
        """
        综合评估（一键获取所有指标）。

        Args:
            prediction: 预测 Mask
            ground_truth: 真值 Mask
            include_boundary: 是否计算边界误差

        Returns:
            包含所有评估指标的字典

        示例:
            >>> results = QualityMetrics.evaluate(pred_mask, gt_mask)
            >>> for k, v in results.items():
            ...     print(f"{k}: {v:.4f}")
        """
        results = {
            "iou": cls.iou(prediction, ground_truth),
            "dice": cls.dice(prediction, ground_truth),
            "precision": cls.precision(prediction, ground_truth),
            "recall": cls.recall(prediction, ground_truth),
            "f1_score": cls.f1_score(prediction, ground_truth),
            "pixel_accuracy": cls.pixel_accuracy(prediction, ground_truth),
        }

        # 面积误差
        area_err = cls.area_error(prediction, ground_truth)
        results.update({
            "area_prediction": area_err["area_prediction"],
            "area_ground_truth": area_err["area_ground_truth"],
            "area_relative_error": area_err["relative_error"],
        })

        # 边界误差
        if include_boundary:
            boundary_err = cls.boundary_error(prediction, ground_truth)
            if boundary_err:
                results.update({
                    "hausdorff_distance": boundary_err["hausdorff_distance"],
                    "avg_boundary_error": boundary_err["avg_boundary_error"],
                })

        return results

    @staticmethod
    def print_report(results: Dict[str, float]) -> None:
        """打印评估报告。"""
        print("\n" + "=" * 50)
        print("       标签质量评估报告")
        print("=" * 50)

        # 核心指标
        print("\n--- 核心指标 ---")
        print(f"  IoU (交并比):     {results.get('iou', 0):.4f}")
        print(f"  Dice 系数:        {results.get('dice', 0):.4f}")
        print(f"  F1 分数:          {results.get('f1_score', 0):.4f}")
        print(f"  像素精度:         {results.get('pixel_accuracy', 0):.4f}")

        # 详细指标
        print("\n--- 详细指标 ---")
        print(f"  精确率:           {results.get('precision', 0):.4f}")
        print(f"  召回率:           {results.get('recall', 0):.4f}")

        # 面积
        print("\n--- 面积统计 ---")
        print(f"  预测面积:         {results.get('area_prediction', 0):.0f} 像素")
        print(f"  真值面积:         {results.get('area_ground_truth', 0):.0f} 像素")
        rel_err = results.get("area_relative_error", 0)
        print(f"  面积相对误差:     {rel_err:.2%}")

        # 边界
        if "hausdorff_distance" in results:
            print("\n--- 边界质量 ---")
            print(f"  Hausdorff 距离:   {results.get('hausdorff_distance', 0):.2f} 像素")
            print(f"  平均边界误差:     {results.get('avg_boundary_error', 0):.2f} 像素")

        # 质量等级
        iou = results.get("iou", 0)
        print("\n--- 质量等级 ---")
        if iou >= 0.9:
            grade = "优秀 (A)"
        elif iou >= 0.75:
            grade = "良好 (B)"
        elif iou >= 0.5:
            grade = "合格 (C)"
        else:
            grade = "需改进 (D)"
        print(f"  综合评级:         {grade}")
        print("=" * 50 + "\n")

    @staticmethod
    def batch_evaluate(
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        批量评估多个样本的平均指标。

        Args:
            predictions: 预测 Mask 列表
            ground_truths: 真值 Mask 列表

        Returns:
            平均指标字典
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测和真值数量不匹配")

        all_results = []
        for pred, gt in zip(predictions, ground_truths):
            r = QualityMetrics.evaluate(pred, gt, include_boundary=False)
            all_results.append(r)

        # 计算平均值
        avg_results = {}
        for key in all_results[0]:
            values = [r[key] for r in all_results]
            avg_results[key] = float(np.mean(values))
            avg_results[f"{key}_std"] = float(np.std(values))

        avg_results["num_samples"] = len(predictions)
        return avg_results
