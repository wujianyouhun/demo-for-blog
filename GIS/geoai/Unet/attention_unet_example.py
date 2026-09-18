from pathlib import Path

from infer_multiclass import main as run_inference


def main() -> None:
    """Attention U-Net 五分类整图推理：下列参数可直接按项目需要修改。"""
    root = Path(__file__).resolve().parent
    model_name = "attention_unet"
    input_image = root / "Data" / "Qilian.tif"
    # 改进版：基于当前样本训练 80 轮，最佳验证集 mIoU 为 0.444；旧流程权重保留不覆盖。
    weights_file = root.parent / "models" / "Unet" / "qilian_multiclass_5class_refined" / model_name / "best.pth"
    output_directory = root / "out" / "qilian_multiclass_5class_refined" / model_name
    tile_size, stride = 256, 192
    target_crs, target_resolution, device = "EPSG:3857", 10.0, "auto"
    run_inference(model_name, input_path=input_image, weights_path=weights_file,
                  output_dir=output_directory, tile_size=tile_size, stride=stride,
                  target_crs=target_crs, target_resolution=target_resolution, device=device)


if __name__ == "__main__":
    main()
