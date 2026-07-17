import torch

from models.deeplabv3_plus import DeepLabV3Plus
from models.unet_plus_plus import UNetPlusPlus


def test_baseline_models_shape_and_gradient():
    for model in (UNetPlusPlus(base_channels=8), DeepLabV3Plus(base_channels=8)):
        model.train()
        x = torch.randn(2, 3, 128, 160, requires_grad=True)
        y = model(x)
        assert y.shape == (2, 2, 128, 160)
        y.mean().backward()
        assert x.grad is not None

