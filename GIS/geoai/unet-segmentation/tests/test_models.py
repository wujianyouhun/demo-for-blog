import torch

from unet_geoai.models import MODEL_NAMES, build_model


def test_all_models_preserve_spatial_shape():
    image = torch.randn(1, 3, 128, 160)
    for name in MODEL_NAMES:
        model = build_model(name, num_classes=6, base_channels=8)
        output = model(image)
        assert output.shape == (1, 6, 128, 160)
        output.mean().backward()
