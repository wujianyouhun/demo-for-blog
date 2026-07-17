import numpy as np
import pytest
from PIL import Image

from utils import SegDataset, calculate_metrics


def test_dataset_requires_strict_pairing(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(); masks.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(images / "a.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(masks / "b.png")
    with pytest.raises(ValueError, match="严格配对"):
        SegDataset(images, masks)


def test_binary_metrics():
    prediction = np.array([[0, 1], [1, 1]])
    target = np.array([[0, 1], [0, 1]])
    metrics = calculate_metrics(prediction, target)
    assert metrics["iou"] == pytest.approx(2 / 3)
    assert metrics["dice"] == pytest.approx(0.8)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0

