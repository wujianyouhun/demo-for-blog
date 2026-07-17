import numpy as np

from unet_geoai.metrics import boundary_f1, confusion_matrix, metrics_from_confusion


def test_perfect_metrics_are_one():
    target = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    matrix = confusion_matrix(target, target, 3)
    metrics = metrics_from_confusion(matrix)
    assert metrics["miou"] == 1.0
    assert metrics["mdice"] == 1.0
    assert boundary_f1(target, target, 3) == 1.0
