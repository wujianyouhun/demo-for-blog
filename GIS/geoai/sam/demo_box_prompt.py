"""
演示脚本 2: 框提示标注

展示如何使用 SAM 模型通过框提示（Box Prompt）
对遥感影像进行交互式分割标注。

框提示是遥感场景最常用的标注方式，
因为建筑轮廓清晰、目标边界明确、精度高于单点。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai_sam import SAMWrapper, MaskPostProcessor, MaskVectorizer


def main():
    # ================================================================
    # 配置参数
    # ================================================================
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "西安19级.tif")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "box_prompt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_TYPE = "vit_l"  # 6GB 显存推荐 vit_l
    SAM_VERSION = "sam1"

    print("=" * 60)
    print("  GeoAI SAM - 框提示标注演示")
    print("=" * 60)
    print(f"影像路径: {IMAGE_PATH}")
    print(f"模型类型: {MODEL_TYPE} ({SAM_VERSION})")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # ================================================================
    # Step 1: 初始化并加载影像
    # ================================================================
    print("[Step 1] 初始化 SAM 模型并加载影像...")
    sam = SAMWrapper(
        model_type=MODEL_TYPE,
        sam_version=SAM_VERSION,
        automatic=False,
    )
    sam.set_image(IMAGE_PATH)
    print()

    # ================================================================
    # Step 2: 单框提示标注
    # ================================================================
    print("[Step 2] 单框提示标注...")
    print("  边界框: [xmin=100, ymin=120, xmax=500, ymax=480]")

    # 坐标格式: [xmin, ymin, xmax, ymax]
    box = [100, 120, 500, 480]

    masks_single = sam.generate_masks_by_box(
        box=box,
        multimask_output=True,
    )

    sam.save_masks(os.path.join(OUTPUT_DIR, "single_box_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "single_box_result.png"),
        title="单框提示分割结果",
    )
    print()

    # ================================================================
    # Step 3: 多框批量标注
    # ================================================================
    print("[Step 3] 多框批量标注...")

    boxes = [
        [100, 120, 500, 480],   # 建筑物区域 1
        [600, 300, 900, 700],   # 建筑物区域 2
    ]

    print(f"  共 {len(boxes)} 个框:")
    for i, b in enumerate(boxes):
        print(f"    框 {i+1}: [{b[0]}, {b[1]}, {b[2]}, {b[3]}]")

    masks_multi = sam.generate_masks_by_boxes(
        boxes=boxes,
        multimask_output=True,
    )

    sam.save_masks(os.path.join(OUTPUT_DIR, "multi_box_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "multi_box_result.png"),
        title="多框批量分割结果",
    )
    print()

    # ================================================================
    # Step 4: 精细框标注 + 后处理
    # ================================================================
    print("[Step 4] 精细框标注 + 后处理...")

    # 对单个目标使用更精细的框
    fine_box = [200, 200, 400, 400]
    masks_fine = sam.generate_masks_by_box(box=fine_box)

    # 后处理流程
    mask = sam.masks
    if mask is not None and isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[0]

        # 使用默认后处理流程
        clean_mask = MaskPostProcessor.default_pipeline(
            mask,
            min_size=100,
            fill_holes_flag=True,
            smooth_sigma=1.0,
            opening_radius=2,
            closing_radius=2,
        )

        # 保存后处理结果
        processor = MaskPostProcessor()
        processor.load(clean_mask)
        processor.save(
            os.path.join(OUTPUT_DIR, "postprocessed_mask.tif"),
            reference_image=IMAGE_PATH,
        )
        processor.visualize(
            save_path=os.path.join(OUTPUT_DIR, "postprocess_comparison.png"),
        )

        stats = processor.get_statistics()
        print(f"\n后处理统计:")
        print(f"  目标数量: {stats['num_objects']}")
        print(f"  覆盖面积: {stats['foreground_pixels']} 像素")
    print()

    # ================================================================
    # Step 5: 矢量化并保存
    # ================================================================
    print("[Step 5] Mask 矢量化...")

    if mask is not None:
        vectorizer = MaskVectorizer()

        # 使用处理后的 mask
        target_mask = clean_mask if 'clean_mask' in dir() else mask

        gdf = vectorizer.vectorize(
            target_mask,
            reference_image=IMAGE_PATH,
            min_area=50,
        )

        # 添加分类属性
        if vectorizer.get_gdf() is not None:
            vectorizer.add_attribute(
                "category",
                ["building"] * vectorizer.get_polygon_count(),
            )

        # 保存为多种格式
        vectorizer.save(
            os.path.join(OUTPUT_DIR, "box_prompt_polygons.geojson"),
        )
        vectorizer.save(
            os.path.join(OUTPUT_DIR, "box_prompt_polygons.gpkg"),
        )

        print(f"  多边形数量: {vectorizer.get_polygon_count()}")
        print(f"  总面积: {vectorizer.get_total_area():.0f} 像素")
    print()

    # ================================================================
    # Step 6: 框提示 + 点提示联合使用
    # ================================================================
    print("[Step 6] 框 + 点联合提示（进阶用法）...")
    print("  先用框大致定位，再用点精细调整")

    # 第一步: 框提示获取初始 mask
    init_box = [150, 150, 450, 450]
    masks_init = sam.generate_masks_by_box(box=init_box)

    # 第二步: 基于初始结果，使用点提示修正
    # 在遗漏区域添加前景点
    refine_points = [[300, 300], [350, 350]]
    masks_refined = sam.generate_masks_by_points(
        points=refine_points,
        point_labels=[1, 1],
    )

    sam.save_masks(os.path.join(OUTPUT_DIR, "refined_mask.tif"))
    sam.show_masks(
        save_path=os.path.join(OUTPUT_DIR, "refined_result.png"),
        title="框+点联合提示分割结果",
    )
    print()

    # ================================================================
    # 输出摘要
    # ================================================================
    print("=" * 60)
    print("  框提示标注完成!")
    print("=" * 60)
    print(f"输出文件目录: {OUTPUT_DIR}")
    print("  - single_box_mask.tif       (单框 Mask)")
    print("  - multi_box_mask.tif        (多框 Mask)")
    print("  - postprocessed_mask.tif    (后处理 Mask)")
    print("  - refined_mask.tif          (联合提示 Mask)")
    print("  - box_prompt_polygons.geojson  (GeoJSON)")
    print("  - box_prompt_polygons.gpkg     (GeoPackage)")


if __name__ == "__main__":
    main()
