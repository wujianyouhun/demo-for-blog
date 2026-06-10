"""
演示脚本 1: 点提示标注

展示如何使用 SAM 模型通过点提示（Point Prompt）
对遥感影像进行交互式分割标注。

适用场景: 建筑物、水体、光伏板、道路节点等目标提取
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai_sam import SAMWrapper, MaskPostProcessor, MaskVectorizer


def main():
    # ================================================================
    # 配置参数
    # ================================================================
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "西安19级.tif")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "point_prompt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # SAM 模型配置
    MODEL_TYPE = "vit_l"  # vit_b / vit_l / vit_h (6GB 显存推荐 vit_l)
    SAM_VERSION = "sam1"  # sam1 / sam2 / sam3

    print("=" * 60)
    print("  GeoAI SAM - 点提示标注演示")
    print("=" * 60)
    print(f"影像路径: {IMAGE_PATH}")
    print(f"模型类型: {MODEL_TYPE} ({SAM_VERSION})")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # ================================================================
    # Step 1: 初始化 SAM 模型
    # ================================================================
    print("[Step 1] 初始化 SAM 模型...")
    sam = SAMWrapper(
        model_type=MODEL_TYPE,
        sam_version=SAM_VERSION,
        automatic=False,
    )
    print()

    # ================================================================
    # Step 2: 加载遥感影像
    # ================================================================
    print("[Step 2] 加载遥感影像...")
    sam.set_image(IMAGE_PATH)
    print()

    # ================================================================
    # Step 3: 单点提示标注
    # ================================================================
    print("[Step 3] 单点提示标注...")
    print("  提示点: [750, 370]")

    masks_single = sam.generate_masks_by_points(
        points=[[750, 370]],
    )

    # 保存结果
    sam.save_masks(os.path.join(OUTPUT_DIR, "single_point_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "single_point_result.png"),
        title="单点提示分割结果",
    )
    print()

    # ================================================================
    # Step 4: 多点提示标注（全前景）
    # ================================================================
    print("[Step 4] 多点提示标注（全前景）...")
    print("  前景点: [500, 375], [1125, 625]")

    masks_multi = sam.generate_masks_by_points(
        points=[
            [500, 375],
            [1125, 625],
        ],
        point_labels=[1, 1],  # 全前景
    )

    sam.save_masks(os.path.join(OUTPUT_DIR, "multi_point_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "multi_point_result.png"),
        title="多点提示分割结果 (全前景)",
    )
    print()

    # ================================================================
    # Step 5: 正负样本联合提示（前景 + 背景）
    # ================================================================
    print("[Step 5] 正负样本联合提示...")
    print("  前景点: [750, 370] (建筑物)")
    print("  背景点: [1125, 625] (树木/非目标)")

    masks_mixed = sam.generate_masks_by_points(
        points=[
            [750, 370],   # 建筑物 - 前景
            [1125, 625],  # 树木 - 背景（排除）
        ],
        point_labels=[1, 0],  # 1=前景, 0=背景
    )

    sam.save_masks(os.path.join(OUTPUT_DIR, "mixed_prompt_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "mixed_prompt_result.png"),
        title="正负样本联合提示分割结果",
    )
    print()

    # ================================================================
    # Step 6: Mask 后处理
    # ================================================================
    print("[Step 6] Mask 后处理...")

    # 获取最后的 mask
    mask = sam.masks
    if mask is not None and isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[0]

        processor = MaskPostProcessor()
        clean_mask = (
            processor
            .load(mask)
            .remove_small_objects(min_size=100)
            .fill_holes()
            .smooth(sigma=1.0)
            .get_result()
        )

        # 保存后处理结果
        processor.save(
            os.path.join(OUTPUT_DIR, "postprocessed_mask.tif"),
            reference_image=IMAGE_PATH,
        )

        # 可视化对比
        processor.visualize(
            save_path=os.path.join(OUTPUT_DIR, "postprocess_comparison.png"),
        )

        # 打印统计
        stats = processor.get_statistics()
        print(f"\n后处理统计:")
        print(f"  目标数量: {stats['num_objects']}")
        print(f"  覆盖面积: {stats['foreground_pixels']} 像素")
        print(f"  覆盖率: {stats['coverage_ratio']:.4f}")
    print()

    # ================================================================
    # Step 7: Mask → Polygon 矢量化
    # ================================================================
    print("[Step 7] Mask 矢量化...")

    if mask is not None:
        vectorizer = MaskVectorizer()
        gdf = vectorizer.vectorize(
            clean_mask if 'clean_mask' in dir() else mask,
            reference_image=IMAGE_PATH,
            min_area=50,
        )

        # 保存为 GeoJSON
        output_geojson = vectorizer.save(
            os.path.join(OUTPUT_DIR, "point_prompt_polygons.geojson")
        )

        print(f"  多边形数量: {vectorizer.get_polygon_count()}")
        print(f"  总面积: {vectorizer.get_total_area():.0f} 像素")
    print()

    # ================================================================
    # 输出摘要
    # ================================================================
    print("=" * 60)
    print("  点提示标注完成!")
    print("=" * 60)
    print(f"输出文件目录: {OUTPUT_DIR}")
    print("  - single_point_mask.tif     (单点提示 Mask)")
    print("  - multi_point_mask.tif      (多点提示 Mask)")
    print("  - mixed_prompt_mask.tif     (正负联合 Mask)")
    print("  - postprocessed_mask.tif    (后处理 Mask)")
    print("  - point_prompt_polygons.geojson (矢量结果)")
    print("  - *.png                     (可视化图片)")


if __name__ == "__main__":
    main()
