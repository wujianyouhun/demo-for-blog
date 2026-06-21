"""测试 - 模型构建"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch


class TestModels:
    def test_build_deeplabv3p_resnet50(self):
        from geoai_core.models import build_model
        model = build_model("deeplabv3p_resnet50")
        assert model is not None
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_forward_pass(self):
        from geoai_core.models import build_model
        model = build_model("deeplabv3p_resnet50")
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 6, 64, 64)

    def test_build_mobilenet(self):
        from geoai_core.models import build_model
        model = build_model("deeplabv3p_mobilenet")
        assert model is not None

    def test_invalid_model(self):
        from geoai_core.models import build_model
        with pytest.raises(ValueError):
            build_model("nonexistent_model")


class TestDataset:
    def test_dataset_class(self):
        from geoai_core.dataset import LandCoverDataset
        assert LandCoverDataset is not None

    def test_augmentation(self):
        from geoai_core.dataset import get_augmentation
        aug = get_augmentation()
        assert aug is not None
