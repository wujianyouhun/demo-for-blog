"""
快速入门示例

最简单的 SAM 遥感标注流程:
    1. 加载影像
    2. 框提示分割
    3. 后处理
    4. 导出 Polygon

用法:
    conda activate geoai
    python quick_start.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai_sam import SAMWrapper, MaskPostProcessor, MaskVectorizer


def main():
    # 影像路径
    image_path = os.path.join(os.path.dirname(__file__), "西安19级.tif")
    output_dir = os.path.join(os.path.dirname(__file__), "output", "quick_start")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("  GeoAI SAM 快速入门")
    print("=" * 50)

    # --------------------------------------------------
    # Step 1: 加载 SAM 模型和影像
    # --------------------------------------------------
    print("\n[1/4] 加载模型和影像...")
    sam = SAMWrapper(model_type="vit_l", automatic=False)
    sam.set_image(image_path)

    # --------------------------------------------------
    # Step 2: 框提示分割
    # --------------------------------------------------
    print("\n[2/4] 框提示分割...")
    # 坐标格式: [xmin, ymin, xmax, ymax]
    masks = sam.generate_masks_by_box(box=[100, 120, 500, 480])
    sam.save_masks(os.path.join(output_dir, "raw_mask.tif"))

    # --------------------------------------------------
    # Step 3: 后处理
    # --------------------------------------------------
    print("\n[3/4] Mask 后处理...")
    import numpy as np
    mask = sam.masks
    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[0]

        clean_mask = MaskPostProcessor.default_pipeline(
            mask,
            min_size=200,
            fill_holes_flag=True,
            smooth_sigma=1.5,
        )
    else:
        print("  未获取到 Mask，跳过后处理")
        clean_mask = None

    # --------------------------------------------------
    # Step 4: 矢量化导出
    # --------------------------------------------------
    print("\n[4/4] 矢量化导出...")
    if clean_mask is not None and clean_mask.sum() > 0:
        vectorizer = MaskVectorizer()
        gdf = vectorizer.vectorize(
            clean_mask,
            reference_image=image_path,
            min_area=50,
        )
        if len(gdf) > 0:
            output_path = vectorizer.save(
                os.path.join(output_dir, "buildings.geojson")
            )
            print(f"\n导出完成!")
            print(f"  多边形数量: {vectorizer.get_polygon_count()}")
            print(f"  输出文件: {output_path}")
        else:
            print("  矢量化未产生多边形（框区域可能无显著目标），可尝试调整框坐标")
    else:
        print("  跳过矢量化（Mask 为空）")

    print("\n" + "=" * 50)
    print("  完成! 查看 output/quick_start/ 目录")
    print("=" * 50)


if __name__ == "__main__":
    main()
