"""
演示脚本 3: 文本提示标注（Grounded-SAM）

展示如何使用 GroundingDINO + SAM 通过文本提示
自动检测并分割遥感影像中的目标。

工作流: 文本提示 → GroundingDINO 检测 → 生成 BBox → SAM 精确分割 → Mask

这是 GeoAI 中最有价值的能力之一，可实现：
- 零样本目标分割（无需训练数据）
- 自动化标注（大幅减少人工操作）
- 语义理解（通过自然语言描述目标）
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai_sam import GroundedSAMWrapper, MaskPostProcessor, MaskVectorizer


def main():
    # ================================================================
    # 配置参数
    # ================================================================
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "西安19级.tif")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "text_prompt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SAM_MODEL_TYPE = "vit_l"  # 6GB 显存推荐 vit_l
    GROUNDINGDINO_MODEL = "GroundingDINO_SwinT"  # SwinT 更轻量

    print("=" * 60)
    print("  GeoAI SAM - 文本提示标注 (Grounded-SAM) 演示")
    print("=" * 60)
    print(f"影像路径: {IMAGE_PATH}")
    print(f"SAM 模型: {SAM_MODEL_TYPE}")
    print(f"GroundingDINO: {GROUNDINGDINO_MODEL}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # ================================================================
    # Step 1: 初始化 Grounded-SAM
    # ================================================================
    print("[Step 1] 初始化 Grounded-SAM...")
    gsam = GroundedSAMWrapper(
        sam_model_type=SAM_MODEL_TYPE,
        groundingdino_model=GROUNDINGDINO_MODEL,
        box_threshold=0.25,
        text_threshold=0.25,
    )
    gsam.set_image(IMAGE_PATH)
    print()

    # ================================================================
    # Step 2: 建筑物提取
    # ================================================================
    print("[Step 2] 文本提示: 'building' (建筑物提取)...")

    masks_building = gsam.segment_by_text(
        text_prompt="building",
        box_threshold=0.25,
        text_threshold=0.25,
    )

    # 保存结果
    gsam.save_results(
        mask_path=os.path.join(OUTPUT_DIR, "building_mask.tif"),
        box_path=os.path.join(OUTPUT_DIR, "building_boxes.geojson"),
    )
    print()

    # ================================================================
    # Step 3: 水体提取
    # ================================================================
    print("[Step 3] 文本提示: 'water. lake. river' (水体提取)...")

    masks_water = gsam.segment_by_text(
        text_prompt="water. lake. river",
        box_threshold=0.2,
        text_threshold=0.2,
    )

    gsam.save_results(
        mask_path=os.path.join(OUTPUT_DIR, "water_mask.tif"),
        box_path=os.path.join(OUTPUT_DIR, "water_boxes.geojson"),
    )
    print()

    # ================================================================
    # Step 4: 道路提取
    # ================================================================
    print("[Step 4] 文本提示: 'road. highway' (道路提取)...")

    masks_road = gsam.segment_by_text(
        text_prompt="road. highway",
        box_threshold=0.3,
        text_threshold=0.25,
    )

    gsam.save_results(
        mask_path=os.path.join(OUTPUT_DIR, "road_mask.tif"),
    )
    print()

    # ================================================================
    # Step 5: CLIP 增强分割
    # ================================================================
    print("[Step 5] CLIP 增强文本分割...")

    masks_clip = gsam.segment_by_text_with_clip(
        text_prompt="building",
        clip_model="ViT-B-32",
    )
    print()

    # ================================================================
    # Step 6: 后处理与矢量化
    # ================================================================
    print("[Step 6] 建筑物 Mask 后处理与矢量化...")

    if masks_building is not None and isinstance(masks_building, np.ndarray):
        mask = masks_building
        if mask.ndim == 3:
            mask = mask[0]

        # 后处理
        clean_mask = MaskPostProcessor.default_pipeline(
            mask,
            min_size=200,
            fill_holes_flag=True,
            smooth_sigma=1.5,
        )

        # 矢量化
        vectorizer = MaskVectorizer()
        gdf = vectorizer.vectorize(
            clean_mask,
            reference_image=IMAGE_PATH,
            min_area=100,
        )

        # 添加属性
        if vectorizer.get_gdf() is not None:
            vectorizer.add_attribute(
                "category",
                ["building"] * vectorizer.get_polygon_count(),
            )
            vectorizer.add_attribute(
                "source",
                ["grounded_sam"] * vectorizer.get_polygon_count(),
            )

        # 保存矢量结果
        vectorizer.save(
            os.path.join(OUTPUT_DIR, "building_polygons.geojson"),
        )
        vectorizer.save(
            os.path.join(OUTPUT_DIR, "building_polygons.gpkg"),
        )

        print(f"\n建筑物提取统计:")
        print(f"  多边形数量: {vectorizer.get_polygon_count()}")
        print(f"  总面积: {vectorizer.get_total_area():.0f} 像素")

        # 保存后处理后的 mask
        processor = MaskPostProcessor()
        processor.load(clean_mask)
        processor.save(
            os.path.join(OUTPUT_DIR, "building_postprocessed.tif"),
            reference_image=IMAGE_PATH,
        )
    print()

    # ================================================================
    # Step 7: 多类别综合提取
    # ================================================================
    print("[Step 7] 多类别综合提取...")

    categories = {
        "building": "building. house. roof",
        "water": "water. lake. pond",
        "vegetation": "tree. forest. vegetation",
    }

    all_results = {}
    for cat_name, prompt in categories.items():
        print(f"  提取 '{cat_name}': '{prompt}'")
        try:
            masks = gsam.segment_by_text(
                text_prompt=prompt,
                box_threshold=0.25,
                text_threshold=0.25,
            )
            if masks is not None:
                all_results[cat_name] = masks
                print(f"    结果: {masks.shape if isinstance(masks, np.ndarray) else '生成完成'}")
            else:
                print(f"    结果: 未检测到目标")
        except Exception as e:
            print(f"    错误: {e}")

    print(f"\n共提取 {len(all_results)} 个类别")
    print()

    # ================================================================
    # 输出遥感常用提示词参考
    # ================================================================
    print("=" * 60)
    print("  遥感常用文本提示词参考")
    print("=" * 60)

    prompt_categories = {
        "建筑物": ["building", "house", "roof", "construction"],
        "水体": ["water", "lake", "river", "pond", "reservoir"],
        "道路": ["road", "highway", "street", "path"],
        "植被": ["tree", "forest", "vegetation", "grass"],
        "光伏板": ["solar panel", "solar farm", "photovoltaic"],
        "车辆": ["car", "truck", "vehicle", "bus"],
        "农田": ["farmland", "crop field", "agricultural land"],
    }

    for cat, prompts in prompt_categories.items():
        print(f"  {cat}: {', '.join(prompts)}")
    print()

    # ================================================================
    # 输出摘要
    # ================================================================
    print("=" * 60)
    print("  文本提示标注完成!")
    print("=" * 60)
    print(f"输出文件目录: {OUTPUT_DIR}")
    print("  - building_mask.tif         (建筑物 Mask)")
    print("  - water_mask.tif            (水体 Mask)")
    print("  - road_mask.tif             (道路 Mask)")
    print("  - building_polygons.geojson (建筑物矢量)")
    print("  - building_boxes.geojson    (检测框)")


if __name__ == "__main__":
    main()
