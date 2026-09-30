"""从 GeoTIFF 和分类面矢量制作对齐的 256×256 U-Net 多分类样本。

默认输入：Unet/DATA/裁剪.tif 与 Unet/DATA/shp/polygon_rg.shp。
输出：按空间列切分的 train / val / test 影像、标签、裁切矢量、格网和 manifest。
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.warp import Resampling, calculate_default_transform, reproject


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "DATA" / "裁剪.tif"
DEFAULT_VECTOR = ROOT / "DATA" / "shp" / "polygon_rg.shp"
DEFAULT_OUTPUT = ROOT / "dataset" / "qilian_multiclass_5class"


def load_geodataframe(path: Path):
    """延迟导入，依赖未安装时给出适合 Windows/Conda 的错误提示。"""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "缺少 GeoPandas。请先执行：\n"
            "conda activate python12\n"
            "conda install -c conda-forge geopandas pyogrio\n"
            "随后重新运行本脚本。"
        ) from exc
    return gpd.read_file(path), gpd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按影像像元网格裁切影像和分类面，并制作 U-Net 训练样本")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="输入 GeoTIFF")
    parser.add_argument("--vector", type=Path, default=DEFAULT_VECTOR, help="含 class_id 的分类面 Shapefile / GeoPackage")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="输出样本目录")
    parser.add_argument("--class-field", default="class_id", help="矢量中的整数类别字段，默认 class_id")
    parser.add_argument("--class-remap", default="auto",
                        help="类别重编码：auto 自动将 1–5 转为 0–4；none 保持 0–4；或写 1:0,2:1,...")
    parser.add_argument("--tile-size", type=int, default=256, help="方形样本边长（像元），默认 256")
    parser.add_argument("--target-crs", default="EPSG:3857", help="统一输出坐标系，默认 EPSG:3857")
    parser.add_argument("--target-resolution", type=float, default=10.0,
                        help="统一输出像元大小（米），默认 10")
    parser.add_argument("--ignore-value", type=int, default=255, help="未标注和边缘填充标签值，默认 255")
    parser.add_argument("--all-touched", action="store_true", help="栅格化时所有被面触及的像元均赋类")
    parser.add_argument("--overwrite", action="store_true", help="删除同名输出目录后重新生成")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise FileNotFoundError(f"未找到影像：{args.image}")
    if not args.vector.is_file():
        raise FileNotFoundError(f"未找到矢量：{args.vector}")
    if args.tile_size < 16 or args.tile_size % 16:
        raise ValueError("--tile-size 必须不小于 16，且能被 16 整除。")
    if args.target_resolution <= 0:
        raise ValueError("--target-resolution 必须为正数。")
    if not 0 <= args.ignore_value <= 255:
        raise ValueError("--ignore-value 必须位于 0–255。")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录已存在且非空：{args.output_dir}。确认后增加 --overwrite 重新生成。")


def grid_starts(length: int, tile_size: int) -> list[int]:
    """不重叠网格起点；最后一格可超出原图，并在输出中填充为 ignore。"""
    return list(range(0, length, tile_size))


def split_for_column(col: int, total_columns: int) -> str:
    """按连续列划分，避免相邻样本分别进入训练与测试造成空间泄漏。"""
    if total_columns < 3:
        raise ValueError("影像至少需要 3 列样本，才能划分 train / val / test。请减小 tile-size 或扩大研究区。")
    train_columns = max(1, math.floor(total_columns * 0.70))
    test_columns = max(1, math.ceil(total_columns * 0.15))
    val_columns = total_columns - train_columns - test_columns
    if val_columns < 1:
        train_columns = max(1, train_columns - 1)
        val_columns = 1
    if col < train_columns:
        return "train"
    if col < train_columns + val_columns:
        return "val"
    return "test"


def ensure_class_ids(vector, class_field: str, ignore_value: int) -> None:
    if class_field not in vector.columns:
        available = ", ".join(str(name) for name in vector.columns if name != "geometry")
        raise ValueError(f"矢量缺少字段 {class_field!r}。现有字段：{available or '无'}。")
    if vector.crs is None:
        raise ValueError("矢量没有坐标系定义；请先在 GIS 中正确定义其 CRS。")
    if vector.geometry.is_empty.any() or vector.geometry.isna().any():
        raise ValueError("矢量中存在空几何，请先删除或修复。")
    if not vector.geometry.is_valid.all():
        raise ValueError("矢量中存在无效几何，请先在 GIS 中执行“修复几何”。")
    values = vector[class_field]
    if values.isna().any():
        raise ValueError(f"字段 {class_field} 存在空值，请先补全。")
    converted = values.astype(float)
    if not np.equal(converted, np.floor(converted)).all():
        raise ValueError(f"字段 {class_field} 必须是整数类别编码。")
    invalid = [int(value) for value in converted if not (0 <= value <= 254 or int(value) == ignore_value)]
    if invalid:
        raise ValueError(f"class_id 必须位于 0–254 或等于 ignore 值 {ignore_value}；发现：{sorted(set(invalid))}")
    vector[class_field] = converted.astype("int16")


def remap_class_ids(vector, class_field: str, value: str) -> None:
    """在内存中把外部标注编码统一为训练使用的 0–4，绝不回写原始 Shapefile。"""
    original = vector[class_field].astype("int16")
    classes = set(int(item) for item in original.unique())
    if value == "auto":
        if classes.issubset({0, 1, 2, 3, 4}):
            mapping: dict[int, int] = {}
        elif classes.issubset({1, 2, 3, 4, 5}):
            mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        else:
            raise ValueError(
                f"无法自动判断类别编码 {sorted(classes)}。请用 --class-remap 指定，例如 1:0,2:1,3:2,4:3,5:4。"
            )
    elif value.lower() == "none":
        mapping = {}
    else:
        try:
            mapping = {int(source): int(target) for source, target in
                       (pair.split(":", 1) for pair in value.split(","))}
        except (ValueError, IndexError) as exc:
            raise ValueError("--class-remap 格式应为 auto、none 或 1:0,2:1,3:2,4:3,5:4。") from exc
    if mapping:
        missing = classes - set(mapping) - {255}
        if missing:
            raise ValueError(f"类别映射缺少源值：{sorted(missing)}")
        vector["source_class_id"] = original
        vector[class_field] = original.map(lambda item: mapping.get(int(item), int(item))).astype("int16")
        print(f"训练标签类别已在内存中重编码：{mapping}；原始矢量文件未改写。")


def write_raster(path: Path, array: np.ndarray, profile: dict, transform, nodata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(height=array.shape[-2], width=array.shape[-1], transform=transform,
                          count=1 if array.ndim == 2 else array.shape[0], compress="deflate", nodata=nodata)
    with rasterio.open(path, "w", **output_profile) as destination:
        if array.ndim == 2:
            destination.write(array, 1)
        else:
            destination.write(array)


def prepare_source_raster(source_path: Path, output_path: Path, target_crs: str, resolution: float) -> Path:
    """将原影像重投影 / 重采样为 EPSG:3857 的 10 m 像元，并保存为样本制作模板。"""
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError("输入影像没有 CRS，无法安全重投影。")
        transform, width, height = calculate_default_transform(
            source.crs, target_crs, source.width, source.height, *source.bounds,
            resolution=(resolution, resolution),
        )
        profile = source.profile.copy()
        profile.update(crs=target_crs, transform=transform, width=width, height=height,
                       compress="deflate")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            for band in range(1, source.count + 1):
                reproject(
                    source=rasterio.band(source, band), destination=rasterio.band(destination, band),
                    src_transform=source.transform, src_crs=source.crs,
                    dst_transform=transform, dst_crs=target_crs,
                    src_nodata=source.nodata, dst_nodata=source.nodata,
                    resampling=Resampling.bilinear,
                )
    print(f"已生成统一影像：{output_path.name}（{target_crs}，{resolution:g} m，{width}×{height} 像元）")
    return output_path


def prepare_samples(args: argparse.Namespace) -> None:
    validate_args(args)
    # 在创建任何输出前检查 GeoPandas、矢量字段和几何，失败时不留下半成品目录。
    vector, gpd = load_geodataframe(args.vector)
    ensure_class_ids(vector, args.class_field, args.ignore_value)
    remap_class_ids(vector, args.class_field, args.class_remap)
    if args.output_dir.exists() and args.overwrite:
        try:
            shutil.rmtree(args.output_dir)
        except PermissionError as exc:
            raise SystemExit(
                f"无法覆盖 {args.output_dir}：其中的文件正在被 GIS 或其他程序打开。"
                "请关闭对应图层后重试，或使用 --output-dir 指定新目录。"
            ) from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_source = prepare_source_raster(
        args.image, args.output_dir / f"source_{args.target_crs.replace(':', '_')}_{args.target_resolution:g}m.tif",
        args.target_crs, args.target_resolution,
    )

    from shapely.geometry import box

    records: list[dict[str, object]] = []
    grid_records: list[dict[str, object]] = []
    with rasterio.open(prepared_source) as source:
        if vector.crs != source.crs:
            print(f"重投影矢量：{vector.crs} → {source.crs}")
            vector = vector.to_crs(source.crs)
        if source.width < args.tile_size or source.height < args.tile_size:
            print("提示：影像边长小于 tile-size，边缘会按 NoData/ignore 填充。")

        rows = grid_starts(source.height, args.tile_size)
        cols = grid_starts(source.width, args.tile_size)
        profile = source.profile.copy()
        image_nodata = source.nodata if source.nodata is not None else 0
        for row_index, row in enumerate(rows):
            for col_index, col in enumerate(cols):
                split = split_for_column(col_index, len(cols))
                tile_id = f"r{row_index:03d}_c{col_index:03d}"
                window = Window(col, row, args.tile_size, args.tile_size)
                transform = source.window_transform(window)
                valid_width = min(args.tile_size, source.width - col)
                valid_height = min(args.tile_size, source.height - row)
                tile_bounds = rasterio.windows.bounds(window, source.transform)
                tile_geometry = box(*tile_bounds)

                image = source.read(window=window, boundless=True, fill_value=image_nodata)
                image_path = args.output_dir / split / "images" / f"{tile_id}.tif"
                image_profile = profile.copy()
                image_profile.update(dtype=source.dtypes[0])
                write_raster(image_path, image, image_profile, transform, image_nodata)

                selected = vector[vector.geometry.intersects(tile_geometry)].copy()
                selected.geometry = selected.geometry.intersection(tile_geometry)
                selected = selected[~selected.geometry.is_empty].copy()
                vector_path = args.output_dir / split / "vectors" / f"{tile_id}.gpkg"
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                selected.to_file(vector_path, layer="labels", driver="GPKG")

                shapes = [(geometry, int(class_id)) for geometry, class_id in
                          zip(selected.geometry, selected[args.class_field])]
                mask = rasterize(shapes, out_shape=(args.tile_size, args.tile_size), transform=transform,
                                 fill=args.ignore_value, dtype="uint8", all_touched=args.all_touched)
                # 原图外部像元永远是 ignore，不能作为 0 类参与训练。
                if valid_height < args.tile_size:
                    mask[valid_height:, :] = args.ignore_value
                if valid_width < args.tile_size:
                    mask[:, valid_width:] = args.ignore_value
                mask_path = args.output_dir / split / "masks" / f"{tile_id}_mask.tif"
                mask_profile = profile.copy()
                mask_profile.update(dtype="uint8")
                write_raster(mask_path, mask, mask_profile, transform, args.ignore_value)

                grid_records.append({"tile_id": tile_id, "split": split, "row": row_index, "col": col_index,
                                     "valid_width": valid_width, "valid_height": valid_height,
                                     "geometry": tile_geometry})
                records.append({"tile_id": tile_id, "split": split, "image": str(image_path.relative_to(args.output_dir)),
                                "mask": str(mask_path.relative_to(args.output_dir)),
                                "vector": str(vector_path.relative_to(args.output_dir)),
                                "valid_width": valid_width, "valid_height": valid_height,
                                "label_polygons": len(selected)})
                print(f"完成 {split:5} {tile_id}：有效像元 {valid_width}×{valid_height}，标签面 {len(selected)}")

        grid = gpd.GeoDataFrame(grid_records, geometry="geometry", crs=source.crs)
        grid.to_file(args.output_dir / "sample_grid.gpkg", layer="grid", driver="GPKG")

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {name: sum(record["split"] == name for record in records) for name in ("train", "val", "test")}
    print(f"\n样本制作完成：{args.output_dir}")
    print(f"train={summary['train']}，val={summary['val']}，test={summary['test']}")
    print("提示：请在 GIS 中打开 sample_grid.gpkg，检查 train/val/test 是否符合预期且各集合类别齐全。")


if __name__ == "__main__":
    prepare_samples(parse_args())
