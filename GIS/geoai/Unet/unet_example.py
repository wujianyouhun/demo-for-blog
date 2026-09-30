from pathlib import Path

from infer_multiclass import main as run_inference


def main() -> None:
    """经典 U-Net 五分类整图推理：下列参数可直接按项目需要修改。"""
    root = Path(__file__).resolve().parent
    model_name = "unet"
    input_image = root / "DATA" / "Qilian.tif"             # 待推理遥感影像
    # 改进版：基于当前样本训练 80 轮，验证集 mIoU 为 0.416；原 3 轮流程权重未覆盖。
    weights_file = root.parent / "models" / "Unet" / "qilian_multiclass_5class_refined" / model_name / "best.pth"
    output_directory = root / "out" / "qilian_multiclass_5class_refined" / model_name
    tile_size = 256                                           # 与训练样本块一致
    stride = 192                                              # 小于块大小，重叠区域取平均概率
    target_crs = "EPSG:3857"
    target_resolution = 10.0                                 # 单位：米
    device = "auto"                                          # auto、cpu 或 cuda
    run_inference(model_name, input_path=input_image, weights_path=weights_file,
                  output_dir=output_directory, tile_size=tile_size, stride=stride,
                  target_crs=target_crs, target_resolution=target_resolution, device=device)


if __name__ == "__main__":
    main()
