"""
演示脚本 4: Mask 后处理

展示 SAM 输出的 Mask 后处理完整流程，包括:
- 去噪（去除小斑块）
- 孔洞填充
- 开闭运算
- 边界平滑
- 面积过滤
- 形态学操作

后处理是将粗糙的 SAM 输出转化为高质量训练标签的关键步骤。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai_sam import SAMWrapper, MaskPostProcessor, MaskVectorizer, QualityMetrics


def main():
    # ================================================================
    # 配置参数
    # ================================================================
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "西安19级.tif")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "postprocess")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  GeoAI SAM - Mask 后处理演示")
    print("=" * 60)
    print()

    # ================================================================
    # Step 1: 获取 SAM 原始 Mask
    # ================================================================
    print("[Step 1] 使用 SAM 获取原始 Mask...")

    sam = SAMWrapper(model_type="vit_l", automatic=False)
    sam.set_image(IMAGE_PATH)

    # 使用框提示获取初始 mask
    masks = sam.generate_masks_by_box(box=[100, 120, 500, 480])

    # 获取原始 mask
    raw_mask = sam.masks
    if raw_mask is not None and isinstance(raw_mask, np.ndarray):
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[0]
        print(f"  原始 Mask 形状: {raw_mask.shape}")
        print(f"  前景像素: {raw_mask.sum()}")
    print()

    # ================================================================
    # Step 2: 去噪 - 去除小斑块
    # ================================================================
    print("[Step 2] 去噪 - 去除小斑块...")

    processor = MaskPostProcessor()
    processor.load(raw_mask)

    # 查看去噪前的统计
    stats_before = processor.get_statistics()
    print(f"  去噪前:")
    print(f"    目标数量: {stats_before['num_objects']}")
    print(f"    最小面积: {stats_before['min_area']} 像素")
    print(f"    最大面积: {stats_before['max_area']} 像素")

    # 去除小于 200 像素的小斑块
    processor.remove_small_objects(min_size=200)

    stats_after = processor.get_statistics()
    print(f"  去噪后:")
    print(f"    目标数量: {stats_after['num_objects']}")
    print(f"    最小面积: {stats_after['min_area']} 像素")
    print()

    # ================================================================
    # Step 3: 孔洞填充
    # ================================================================
    print("[Step 3] 孔洞填充...")
    processor.fill_holes()
    print()

    # ================================================================
    # Step 4: 开运算（去毛刺）
    # ================================================================
    print("[Step 4] 开运算（去毛刺）...")
    processor.opening(radius=2)
    print()

    # ================================================================
    # Step 5: 闭运算（填裂缝）
    # ================================================================
    print("[Step 5] 闭运算（填裂缝）...")
    processor.closing(radius=3)
    print()

    # ================================================================
    # Step 6: 边界平滑
    # ================================================================
    print("[Step 6] 边界平滑...")
    processor.smooth(sigma=1.5)
    print()

    # ================================================================
    # Step 7: 保存后处理结果
    # ================================================================
    print("[Step 7] 保存后处理结果...")

    clean_mask = processor.get_result()

    processor.save(
        os.path.join(OUTPUT_DIR, "clean_mask.tif"),
        reference_image=IMAGE_PATH,
    )

    # 同时保存原始 mask 用于对比
    np.save(os.path.join(OUTPUT_DIR, "raw_mask.npy"), raw_mask)
    np.save(os.path.join(OUTPUT_DIR, "clean_mask.npy"), clean_mask)
    print()

    # ================================================================
    # Step 8: 可视化对比
    # ================================================================
    print("[Step 8] 可视化对比...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 原始影像
    try:
        import rasterio
        with rasterio.open(IMAGE_PATH) as src:
            img = src.read()
            if img.shape[0] >= 3:
                img_show = np.transpose(img[:3], (1, 2, 0))
                img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
            else:
                img_show = img[0]
            img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
        axes[0, 0].imshow(img_show)
    except Exception:
        axes[0, 0].text(0.5, 0.5, "Image", ha="center", va="center")
    axes[0, 0].set_title("原始影像")
    axes[0, 0].axis("off")

    # SAM 原始 Mask
    axes[0, 1].imshow(raw_mask, cmap="Reds", alpha=0.7)
    axes[0, 1].set_title(f"SAM 原始 Mask ({raw_mask.sum()} 像素)")
    axes[0, 1].axis("off")

    # 后处理 Mask
    axes[0, 2].imshow(clean_mask, cmap="Greens", alpha=0.7)
    axes[0, 2].set_title(f"后处理 Mask ({clean_mask.sum()} 像素)")
    axes[0, 2].axis("off")

    # 叠加对比: 原始
    if 'img_show' in dir():
        axes[1, 0].imshow(img_show)
        axes[1, 0].imshow(raw_mask, cmap="Reds", alpha=0.4)
    axes[1, 0].set_title("原始 Mask 叠加")
    axes[1, 0].axis("off")

    # 叠加对比: 后处理
    if 'img_show' in dir():
        axes[1, 1].imshow(img_show)
        axes[1, 1].imshow(clean_mask, cmap="Greens", alpha=0.4)
    axes[1, 1].set_title("后处理 Mask 叠加")
    axes[1, 1].axis("off")

    # 差异图
    diff = raw_mask.astype(int) - clean_mask.astype(int)
    im = axes[1, 2].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 2].set_title("差异图 (红=移除, 蓝=添加)")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2])

    plt.suptitle("SAM Mask 后处理对比", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "postprocess_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  对比图已保存")
    print()

    # ================================================================
    # Step 9: 操作日志与统计
    # ================================================================
    print("[Step 9] 操作日志...")
    for op in processor.get_operations_log():
        print(f"  {op}")

    # 统计对比
    print(f"\n统计对比:")
    print(f"  原始前景像素: {raw_mask.sum()}")
    print(f"  处理后前景像素: {clean_mask.sum()}")
    print(f"  变化: {clean_mask.sum() - raw_mask.sum():+d} 像素")
    print()

    # ================================================================
    # Step 10: 一键后处理流程（简便方法）
    # ================================================================
    print("[Step 10] 使用默认一键后处理流程...")

    quick_clean = MaskPostProcessor.default_pipeline(
        raw_mask,
        min_size=200,
        fill_holes_flag=True,
        smooth_sigma=1.5,
        opening_radius=2,
        closing_radius=3,
    )

    print(f"  一键处理后前景像素: {quick_clean.sum()}")

    # 矢量化
    vectorizer = MaskVectorizer()
    gdf = vectorizer.vectorize(
        quick_clean,
        reference_image=IMAGE_PATH,
        min_area=100,
    )
    vectorizer.save(os.path.join(OUTPUT_DIR, "final_polygons.geojson"))
    print(f"  最终多边形: {vectorizer.get_polygon_count()} 个")
    print()

    # ================================================================
    # 输出摘要
    # ================================================================
    print("=" * 60)
    print("  Mask 后处理演示完成!")
    print("=" * 60)
    print(f"输出文件目录: {OUTPUT_DIR}")
    print("  - raw_mask.npy              (原始 Mask)")
    print("  - clean_mask.tif/npy        (后处理 Mask)")
    print("  - postprocess_comparison.png (对比图)")
    print("  - final_polygons.geojson    (矢量结果)")


if __name__ == "__main__":
    main()
