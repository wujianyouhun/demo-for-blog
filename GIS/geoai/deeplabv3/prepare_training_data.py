"""核查 image1/image2 的分类面，按名称统一六类并生成空间分列样本。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["road", "built_up", "cropland", "bare_land", "water", "solar_farm"]
CLASS_LABELS = ["道路", "建设用地/居民地", "耕地", "裸地", "水体", "光伏"]
NAME_TO_ID = {"道路": 0, "建设用地": 1, "居民地": 1, "耕地": 2,
              "裸地": 3, "水体": 4, "光伏板": 5, "光伏": 5, "光伏场": 5}
COLORS = np.array([[230, 65, 60], [245, 165, 40], [65, 180, 75],
                   [185, 155, 110], [45, 120, 235], [185, 70, 210]], dtype="uint8")
IGNORE = 255


def counts(mask):
    return [int((mask == i).sum()) for i in range(len(CLASS_NAMES))]


def unified_ids(frame):
    """以中文语义统一编号；绝不直接合并两份不同含义的 class_id。"""
    field = "class_name" if "class_name" in frame else "LB" if "LB" in frame else None
    if field is None:
        raise ValueError("标注缺少 class_name 或 LB 中文类别字段。")
    names = frame[field].fillna("").astype(str).str.strip()
    unknown = sorted(set(names) - set(NAME_TO_ID))
    if unknown:
        raise ValueError(f"存在未配置的类别名称：{unknown}，请补充 NAME_TO_ID。")
    return field, names.map(NAME_TO_ID).to_numpy(dtype="uint8")


def make_labels(frame, shape, transform):
    field, ids = unified_ids(frame)
    mask = np.full(shape, IGNORE, dtype="uint8")
    hits = np.zeros(shape, dtype="uint8")
    for class_id in range(len(CLASS_NAMES)):
        geometries = list(frame.geometry[ids == class_id])
        if not geometries:
            continue
        burn = rasterize([(g, 1) for g in geometries], out_shape=shape,
                         transform=transform, fill=0, dtype="uint8", all_touched=False).astype(bool)
        hits += burn
        mask[burn] = class_id
    # 冲突像元不根据要素顺序决定归属，而是显式排除训练。
    mask[hits > 1] = IGNORE
    return mask, hits, field, ids


def preview(path, image, mask, name, bands, overlay_title="统一类别标注 / annotation overlay"):
    rgb = image.astype("float32")
    for c in range(3):
        valid = rgb[c][np.isfinite(rgb[c])]
        low, high = np.percentile(valid, [2, 98])
        rgb[c] = np.clip((rgb[c] - low) / max(high - low, 1), 0, 1)
    rgb = (np.nan_to_num(rgb).transpose(1, 2, 0) * 255).astype("uint8")
    annotated = mask != IGNORE
    colored = rgb.copy()
    colored[annotated] = (rgb[annotated] * 0.55 + COLORS[mask[annotated]] * 0.45).astype("uint8")
    ratio = min(1, 820 / rgb.shape[1])
    size = (max(1, round(rgb.shape[1] * ratio)), max(1, round(rgb.shape[0] * ratio)))
    left = Image.fromarray(rgb).resize(size)
    right = Image.fromarray(colored).resize(size)
    canvas = Image.new("RGB", (size[0] * 2 + 36, size[1] + 140), "white")
    canvas.paste(left, (12, 44))
    canvas.paste(right, (size[0] + 24, 44))
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{name}: RGB bands {bands} (2-98% display stretch)", font=font, fill="black")
    draw.text((size[0] + 24, 10), overlay_title, font=font, fill="black")
    for i, label in enumerate(CLASS_LABELS):
        x, y = 12 + (i % 3) * (canvas.width // 3), size[1] + 58 + (i // 3) * 32
        draw.rectangle((x, y, x + 18, y + 18), fill=tuple(COLORS[i]))
        draw.text((x + 24, y - 2), f"{i}: {label}", font=font, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_tile(path, data, profile, transform, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 重新构造 profile，避免沿用原图块大小和其他不适用于小切片的配置。
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[-2], width=data.shape[-1],
                       count=1 if data.ndim == 2 else data.shape[0], dtype=data.dtype,
                       crs=profile["crs"], transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(data[None] if data.ndim == 2 else data)


def prepare(input_dir, output_dir, report_dir, bands=(1, 2, 3), scale=10000.,
            tile_size=256, val_column=1, inspect_only=False):
    if tile_size < 32 or val_column < 0 or len(bands) != 3 or min(bands) < 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("tile-size 至少为 32，val-column 非负，RGB 波段及 scale 必须有效。")
    if not inspect_only and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录已非空，请使用新目录：{output_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = dict(class_names=CLASS_NAMES, class_labels=CLASS_LABELS, name_to_id=NAME_TO_ID,
                  bands=list(bands), scale=scale, tile_size=tile_size, scenes=[], manifest=[],
                  split_strategy=f"each scene column {val_column} is validation; other columns train",
                  preprocessing_note="Band names/scaling absent from source metadata: RGB order and scale are assumptions to confirm.",
                  evaluation_note="Adjacent nonoverlapping spatial strips; no independent test region. Validation is preliminary.")
    split_counts = {split: np.zeros(len(CLASS_NAMES), dtype="int64") for split in ("train", "val")}
    for stem in ("image1", "image2"):
        image_path, vector_path = input_dir / f"{stem}.tif", input_dir / f"{stem}.shp"
        vector = gpd.read_file(vector_path)
        if vector.crs is None or vector.geometry.isna().any() or vector.geometry.is_empty.any() or not vector.is_valid.all():
            raise ValueError(f"{stem} 存在缺失 CRS、空几何或无效几何，请先修复。")
        if not vector.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError("标签必须为面要素。")
        with rasterio.open(image_path) as src:
            if src.crs is None or max(bands) > src.count:
                raise ValueError(f"{stem} 缺少 CRS 或所需波段。")
            vector = vector.to_crs(src.crs)
            labels, hits, field, ids = make_labels(vector, src.shape, src.transform)
            selected = src.read(bands)
            valid = (src.read_masks(bands) > 0).all(axis=0) & np.isfinite(selected).all(axis=0)
            labels[~valid] = IGNORE
            ncols = (src.width + tile_size - 1) // tile_size
            if ncols < 2 or val_column >= ncols:
                raise ValueError(f"{stem} 仅 {ncols} 列，无法使用验证列 {val_column}。")
            polygon_classes = []
            for (source_id, label), group in vector.groupby(["class_id", field]):
                polygon_classes.append(dict(source_class_id=int(source_id), source_name=str(label),
                                            unified_id=NAME_TO_ID[str(label).strip()], polygons=len(group)))
            entry = dict(name=stem, image=str(image_path.resolve()), vector=str(vector_path.resolve()),
                         width=src.width, height=src.height, channels=src.count, dtype=src.dtypes[0],
                         crs=str(src.crs), bounds=list(src.bounds), pixel_size=list(src.res), nodata=src.nodata,
                         colorinterp=[v.name for v in src.colorinterp], band_descriptions=list(src.descriptions),
                         vector_count=len(vector), valid_geometries=int(vector.is_valid.sum()),
                         outside_image_polygons=sum(not g.within(box(*src.bounds).buffer(1e-8)) for g in vector.geometry),
                         class_field=field, source_classes=polygon_classes,
                         per_class_pixels=counts(labels), unlabelled_pixels=int(((hits == 0) & valid).sum()),
                         conflict_pixels=int((hits > 1).sum()), invalid_image_pixels=int((~valid).sum()),
                         labelled_fraction=float((labels != IGNORE).mean()),
                         value_percentiles=[np.percentile(b[np.isfinite(b)], [0, 2, 50, 98, 100]).tolist() for b in selected])
            report["scenes"].append(entry)
            preview(report_dir / f"{stem}_review.png", selected, labels, stem, list(bands))
            if inspect_only:
                continue
            write_tile(output_dir / "full_masks" / f"{stem}_mask.tif", labels, src.profile, src.transform, IGNORE)
            for y in range(0, src.height, tile_size):
                for x in range(0, src.width, tile_size):
                    h, w = min(tile_size, src.height - y), min(tile_size, src.width - x)
                    mask = np.full((tile_size, tile_size), IGNORE, dtype="uint8")
                    mask[:h, :w] = labels[y:y+h, x:x+w]
                    if not (mask != IGNORE).any():
                        continue
                    split = "val" if x // tile_size == val_column else "train"
                    name = f"{stem}_r{y // tile_size:03d}_c{x // tile_size:03d}"
                    image = src.read(window=Window(x, y, tile_size, tile_size), boundless=True, fill_value=0)
                    transform = window_transform(Window(x, y, tile_size, tile_size), src.transform)
                    write_tile(output_dir / split / "images" / f"{name}.tif", image, src.profile, transform, src.nodata)
                    write_tile(output_dir / split / "masks" / f"{name}_mask.tif", mask, src.profile, transform, IGNORE)
                    pixels = counts(mask)
                    split_counts[split] += pixels
                    report["manifest"].append(dict(name=name, scene=stem, split=split, x=x, y=y,
                                                   valid_width=w, valid_height=h, per_class_pixels=pixels))
            print(f"{stem}: {len(vector)} 个标注面，有效标注 {entry['labelled_fraction']:.2%}，冲突像元 {entry['conflict_pixels']}", flush=True)
    report["split_class_pixels"] = {key: value.tolist() for key, value in split_counts.items()}
    (report_dir / "inspection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not inspect_only:
        (output_dir / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for split, values in split_counts.items():
            missing = [CLASS_NAMES[i] for i in np.flatnonzero(values == 0)]
            if missing:
                raise ValueError(f"{split} 缺少类别 {missing}，请更换 val-column 或补充样本，不能直接训练。")
        print("样本数量：", {s: sum(m["split"] == s for m in report["manifest"]) for s in split_counts})
        print("训练样本：", output_dir.resolve())
    print("核查报告：", report_dir.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "landcover_6class")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "out" / "training_data_review")
    parser.add_argument("--bands", nargs=3, type=int, default=[1, 2, 3])
    parser.add_argument("--scale", type=float, default=10000.)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--val-column", type=int, default=1)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    prepare(args.input_dir, args.output_dir, args.report_dir, args.bands, args.scale,
            args.tile_size, args.val_column, args.inspect_only)


if __name__ == "__main__":
    main()
