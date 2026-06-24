"""
宝鸡市 SAM 标注脚本

对 E:\data\baoji\宝鸡市\I48E006020\I48E006020.tif 进行 SAM 分割标注。
由于影像尺寸巨大 (46604 x 31069)，采用子区域裁剪策略：
    1. 从原始影像裁取指定窗口
    2. 在子区域上运行 SAM 推理
    3. 后处理 + 矢量化导出

用法:
    python baoji_sam_annotate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai_sam import SAMWrapper, MaskPostProcessor, MaskVectorizer


# ================================================================
# 配置参数
# ================================================================

# 原始影像路径
IMAGE_PATH = r"E:\data\baoji\宝鸡市\I48E006020\I48E006020.tif"

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "baoji")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SAM 模型配置
MODEL_TYPE = "vit_b"   # vit_b (375MB, 快速测试) / vit_l (1.2GB, 推荐) / vit_h (2.5GB, 精度最高)
SAM_VERSION = "sam1"   # sam1 / sam2 / sam3

# 子区域裁剪窗口 (像素坐标: col_off, row_off, width, height)
# 选取影像中心区域 2000x2000 进行标注
WINDOW = (21000, 13000, 2000, 2000)

# 框提示坐标 (相对于裁剪后的子影像)
# [xmin, ymin, xmax, ymax]
BOX_PROMPT = [200, 200, 800, 800]


def crop_subimage(image_path: str, window: tuple, output_path: str) -> str:
    """从大影像中裁取子区域并保存为 GeoTIFF。

    Args:
        image_path: 原始影像路径
        window: (col_off, row_off, width, height) 像素坐标
        output_path: 裁剪后影像保存路径

    Returns:
        裁剪后影像路径
    """
    import rasterio
    from rasterio.windows import Window

    col_off, row_off, win_w, win_h = window

    with rasterio.open(image_path) as src:
        win = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
        data = src.read(window=win)

        profile = src.profile.copy()
        profile.update(
            width=win_w,
            height=win_h,
            transform=rasterio.windows.transform(win, src.transform),
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    print(f"  子影像已保存: {output_path}")
    print(f"  尺寸: {win_w} x {win_h}")
    return output_path


def main():
    print("=" * 60)
    print("  宝鸡市 SAM 遥感标注")
    print("=" * 60)
    print(f"原始影像: {IMAGE_PATH}")
    print(f"模型类型: {MODEL_TYPE} ({SAM_VERSION})")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # ================================================================
    # Step 1: 裁取子区域
    # ================================================================
    print("[Step 1] 裁取子区域...")
    sub_image_path = os.path.join(OUTPUT_DIR, "sub_region.tif")
    crop_subimage(IMAGE_PATH, WINDOW, sub_image_path)
    print()

    # ================================================================
    # Step 2: 初始化 SAM 模型并加载子影像
    # ================================================================
    print("[Step 2] 初始化 SAM 模型并加载子影像...")
    sam = SAMWrapper(
        model_type=MODEL_TYPE,
        sam_version=SAM_VERSION,
        automatic=False,
    )
    sam.set_image(sub_image_path)
    print()

    # ================================================================
    # Step 3: 框提示分割
    # ================================================================
    print(f"[Step 3] 框提示分割...")
    print(f"  边界框: {BOX_PROMPT}")

    masks = sam.generate_masks_by_box(box=BOX_PROMPT)
    sam.save_masks(os.path.join(OUTPUT_DIR, "box_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "box_result.png"),
        title="宝鸡市 - 框提示分割结果",
    )
    print()

    # ================================================================
    # Step 4: 点提示分割 (补充)
    # ================================================================
    print("[Step 4] 点提示分割...")
    point_coords = [500, 500]
    print(f"  提示点: {point_coords}")

    masks_point = sam.generate_masks_by_points(
        points=[point_coords],
        point_labels=[1],
    )
    sam.save_masks(os.path.join(OUTPUT_DIR, "point_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "point_result.png"),
        title="宝鸡市 - 点提示分割结果",
    )
    print()

    # ================================================================
    # Step 5: Mask 后处理
    # ================================================================
    print("[Step 5] Mask 后处理...")
    mask = sam.masks
    if mask is not None and isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[0]

        clean_mask = MaskPostProcessor.default_pipeline(
            mask,
            min_size=200,
            fill_holes_flag=True,
            smooth_sigma=1.5,
            opening_radius=2,
            closing_radius=3,
        )

        processor = MaskPostProcessor()
        processor.load(clean_mask)
        processor.save(
            os.path.join(OUTPUT_DIR, "postprocessed_mask.tif"),
            reference_image=sub_image_path,
        )
        processor.visualize(
            save_path=os.path.join(OUTPUT_DIR, "postprocess_comparison.png"),
        )

        stats = processor.get_statistics()
        print(f"  目标数量: {stats['num_objects']}")
        print(f"  覆盖面积: {stats['foreground_pixels']} 像素")
        print(f"  覆盖率: {stats['coverage_ratio']:.4f}")
    else:
        print("  未获取到 Mask，跳过后处理")
        clean_mask = None
    print()

    # ================================================================
    # Step 6: 矢量化导出
    # ================================================================
    print("[Step 6] 矢量化导出...")
    if clean_mask is not None and clean_mask.sum() > 0:
        vectorizer = MaskVectorizer()
        gdf = vectorizer.vectorize(
            clean_mask,
            reference_image=sub_image_path,
            min_area=50,
        )

        if len(gdf) > 0:
            vectorizer.save(os.path.join(OUTPUT_DIR, "polygons.geojson"))
            vectorizer.save(os.path.join(OUTPUT_DIR, "polygons.gpkg"))

            print(f"  多边形数量: {vectorizer.get_polygon_count()}")
            print(f"  总面积: {vectorizer.get_total_area():.0f} 像素")
        else:
            print("  矢量化未产生多边形，可尝试调整框坐标或后处理参数")
    else:
        print("  跳过矢量化 (Mask 为空)")
    print()

    # ================================================================
    # 输出摘要
    # ================================================================
    print("=" * 60)
    print("  宝鸡市 SAM 标注完成!")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print("  - sub_region.tif             (裁剪子影像)")
    print("  - box_mask.tif               (框提示 Mask)")
    print("  - box_result.png             (框提示可视化)")
    print("  - point_mask.tif             (点提示 Mask)")
    print("  - point_result.png           (点提示可视化)")
    print("  - postprocessed_mask.tif     (后处理 Mask)")
    print("  - postprocess_comparison.png (后处理对比)")
    print("  - polygons.geojson           (矢量结果 GeoJSON)")
    print("  - polygons.gpkg              (矢量结果 GeoPackage)")


if __name__ == "__main__":
    main()
