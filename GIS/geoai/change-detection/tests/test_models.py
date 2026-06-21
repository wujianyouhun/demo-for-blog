"""测试 - 模型构建"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch


class TestModels:
    """测试模型构建与前向传播"""

    def test_siamese_unet_build(self):
        from cdd.models import build_model
        model = build_model("siamese_unet", in_channels=3)
        assert model is not None
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_siamese_unet_forward(self):
        from cdd.models import build_model
        model = build_model("siamese_unet", in_channels=3)
        model.eval()
        a = torch.randn(2, 3, 64, 64)
        b = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(a, b)
        assert "out" in out
        assert out["out"].shape[0] == 2
        assert out["out"].shape[1] == 2  # 二分类

    def test_bit_build(self):
        from cdd.models import build_model
        model = build_model("bit", in_channels=3)
        assert model is not None

    def test_bit_forward(self):
        from cdd.models import build_model
        model = build_model("bit", in_channels=3)
        model.eval()
        a = torch.randn(2, 3, 64, 64)
        b = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(a, b)
        assert out["out"].shape == (2, 2, 64, 64)

    def test_invalid_model(self):
        from cdd.models import build_model
        with pytest.raises(ValueError):
            build_model("nonexistent_model")


class TestMetrics:
    """测试评价指标"""

    def test_perfect_prediction(self):
        from cdd.metrics import ChangeMetrics
        import numpy as np
        pred = np.array([1, 1, 0, 0])
        gt = np.array([1, 1, 0, 0])
        m = ChangeMetrics.compute(pred, gt)
        assert m["f1"] == 1.0
        assert m["oa"] == 1.0
        assert m["iou"] == 1.0

    def test_all_wrong(self):
        from cdd.metrics import ChangeMetrics
        import numpy as np
        pred = np.array([0, 0, 1, 1])
        gt = np.array([1, 1, 0, 0])
        m = ChangeMetrics.compute(pred, gt)
        assert m["f1"] == 0.0

    def test_partial_match(self):
        from cdd.metrics import ChangeMetrics
        import numpy as np
        pred = np.array([1, 1, 1, 0])
        gt = np.array([1, 1, 0, 0])
        m = ChangeMetrics.compute(pred, gt)
        assert 0 < m["f1"] < 1
        assert m["precision"] < 1.0  # 有一个 FP
        assert m["recall"] == 1.0     # 全部 TP 找到

    def test_confusion_matrix(self):
        from cdd.metrics import ChangeMetrics
        import numpy as np
        pred = np.array([1, 0, 1, 0])
        gt = np.array([1, 0, 0, 1])
        cm = ChangeMetrics.confusion_matrix(pred, gt)
        assert cm.shape == (2, 2)
        assert cm[1, 1] == 1  # TP
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN


class TestDataset:
    """测试数据集"""

    def test_dataset_class(self):
        from cdd.dataset import BiTemporalDataset
        # 仅测试类可以实例化
        assert BiTemporalDataset is not None

    def test_augmentation(self):
        from cdd.dataset import get_augmentation
        aug = get_augmentation()
        assert aug is not None
