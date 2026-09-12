from pathlib import Path
from unet_pipeline import main as run_inference


def main() -> None:
    """Swin-UNet 启动参数；窗口注意力适合高分辨率影像。"""
    root = Path(__file__).resolve().parent
    run_inference("swin_unet", default_task="both", default_data_dir=root / "data",
                  default_out_dir=root / "out", default_tile_size=512,
                  default_stride=384, default_threshold=0.50, default_device="auto")


if __name__ == "__main__":
    main()
