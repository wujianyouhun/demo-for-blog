"""无需下载权重的回归检查：窗口覆盖、NoData、地理参考与样本对齐。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import torch
from shapely.geometry import box
from torch.utils.data import DataLoader, TensorDataset

from common import to_tensor
from generate_demo_data import write_pair
from remote_sensing_inference import predict
from train_remote_sensing import TileDataset, run_epoch
from prepare_training_data import make_labels, unified_ids


class ConstantModel(torch.nn.Module):
    def forward(self, image):
        result = torch.zeros((image.shape[0], 3, *image.shape[-2:]), device=image.device)
        result[:, 1] = 10
        return {"out": result}


class WorkflowTests(unittest.TestCase):
    def test_annotation_semantics_and_conflicts(self):
        # 相同 source id=5 在两份数据中代表不同语义，必须按中文名称重编码。
        frame = gpd.GeoDataFrame({"class_id": [5, 5], "class_name": ["水体", "光伏板"]},
                                 geometry=[box(0, 0, 2, 2), box(1, 0, 3, 2)], crs="EPSG:32647")
        _, ids = unified_ids(frame)
        np.testing.assert_array_equal(ids, [4, 5])
        mask, hits, _, _ = make_labels(frame, (3, 4), from_origin(0, 3, 1, 1))
        self.assertEqual(int((hits > 1).sum()), 2)
        self.assertTrue((mask[hits > 1] == 255).all())
        self.assertEqual(int((mask == 4).sum()), 2)
        self.assertEqual(int((mask == 5).sum()), 2)
        self.assertTrue((mask[hits == 0] == 255).all())
        frame["class_name"] = ["未定义类别", "光伏板"]
        with self.assertRaisesRegex(ValueError, "未配置"):
            unified_ids(frame)

    def test_per_class_validation_metrics(self):
        target = torch.tensor([[[1, 1, 0], [255, 2, 1]]])
        loader = DataLoader(TensorDataset(torch.zeros(1, 3, 2, 3), target))
        loss, miou, metrics = run_epoch(ConstantModel().eval(), loader, torch.device("cpu"), 3,
                                        return_details=True)
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(miou, 0.2)
        self.assertEqual(metrics["confusion_matrix"], [[0, 1, 0], [0, 3, 0], [0, 1, 0]])
        self.assertAlmostEqual(metrics["per_class_iou"][1], 0.6)

    def test_window_coverage_and_georeference(self):
        # 不同尺寸覆盖多窗口、尾部非整块及比 halo 更小的影像。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for height, width in ((101, 147), (5, 7)):
                image = np.ones((3, height, width), dtype="uint8") * 100
                image[:, 0, 0] = 0
                path, output = root / "source.tif", root / "result.tif"
                transform = from_origin(500000, 4200000, 10, 10)
                with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                                   count=3, dtype="uint8", crs="EPSG:32647", transform=transform, nodata=0) as dst:
                    dst.write(image)
                with patch("remote_sensing_inference.load_checkpoint", return_value=(
                    ConstantModel(), dict(bands=[1, 2, 3], scale=255.0, class_names=["a", "b", "c"])
                )):
                    counts = predict(path, output, root / "unused.pth", "cpu", 64, 8)
                with rasterio.open(output) as dst:
                    expected = np.ones((height, width), dtype="uint8")
                    expected[0, 0] = 255
                    np.testing.assert_array_equal(dst.read(1), expected)
                    self.assertEqual(dst.transform, transform)
                    self.assertEqual(dst.crs.to_epsg(), 32647)
                    self.assertEqual(dst.nodata, 255)
                    self.assertEqual(dst.shape, (height, width))
                self.assertEqual(int(counts.sum()), height * width - 1)

    def test_dataset_and_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image, mask = root / "train/images/one.tif", root / "train/masks/one_mask.tif"
            write_pair(image, mask, np.random.default_rng(42), 32, 32, 500000)
            dataset = TileDataset(root, "train", [1, 2, 3], 255.0, 32, 3)
            tensor, labels = dataset[0]
            self.assertEqual(tensor.shape, (3, 32, 32))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertTrue((labels[:4, :4] == 255).all())
            with rasterio.open(mask, "r+") as dst:
                dst.transform = from_origin(500001, 4200000, 10, 10)
            with self.assertRaisesRegex(ValueError, "像元网格"):
                dataset[0]
        for scale in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                to_tensor(np.zeros((3, 2, 2)), scale)


if __name__ == "__main__":
    unittest.main()
