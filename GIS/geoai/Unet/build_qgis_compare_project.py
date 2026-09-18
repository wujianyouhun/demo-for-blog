"""整理六模型推理结果命名，并生成 QGIS 对比工程。"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from uuid import uuid4
import xml.etree.ElementTree as ET
from xml.dom import minidom

import rasterio


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "out" / "qilian_multiclass_5class_refined"
PROJECT_ROOT = SOURCE_ROOT / "qgis_compare"

MODELS = [
    ("unet", "U-Net"),
    ("resunet", "ResUNet"),
    ("attention_unet", "Attention U-Net"),
    ("unetpp", "U-Net++"),
    ("transunet", "TransUNet"),
    ("swin_unet", "Swin-UNet"),
]

GROUPS = {
    "classification": "01_分类结果_classification",
    "building": "02_建筑提取_building",
    "vegetation": "03_植被耕地提取_vegetation",
}

CLASS_COLORS = {
    "0": ("道路", "220,50,47,255"),
    "1": ("居民地/建筑", "255,165,0,255"),
    "2": ("耕地/植被", "76,175,80,255"),
    "3": ("裸地", "156,112,69,255"),
    "4": ("光伏场", "33,150,243,255"),
}
BINARY_COLORS = {
    "0": ("背景", "0,0,0,0"),
    "1": ("目标", "255,165,0,255"),
}
VEGETATION_COLORS = {
    "0": ("背景", "0,0,0,0"),
    "1": ("耕地/植被", "76,175,80,255"),
}


def copy_named_outputs() -> list[dict[str, str | Path]]:
    folders = {key: PROJECT_ROOT / key for key in GROUPS}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, str | Path]] = []
    for model, display in MODELS:
        items = [
            (
                "classification",
                SOURCE_ROOT / model / "classification.tif",
                folders["classification"] / f"{model}_classification.tif",
                f"{display} 分类",
            ),
            (
                "building",
                SOURCE_ROOT / model / "derived_extracts" / "building_mask.tif",
                folders["building"] / f"{model}_building.tif",
                f"{display} 建筑",
            ),
            (
                "vegetation",
                SOURCE_ROOT / model / "derived_extracts" / "vegetation_mask.tif",
                folders["vegetation"] / f"{model}_vegetation.tif",
                f"{display} 植被_耕地",
            ),
        ]
        for group, src, dst, layer_name in items:
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
            outputs.append({"group": group, "model": model, "path": dst, "name": layer_name})
    return outputs


def add_srs(parent: ET.Element) -> None:
    srs = ET.SubElement(parent, "srs")
    spatialrefsys = ET.SubElement(srs, "spatialrefsys")
    values = [
        ("wkt", ""),
        ("proj4", "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs"),
        ("srsid", "3857"),
        ("srid", "3857"),
        ("authid", "EPSG:3857"),
        ("description", "WGS 84 / Pseudo-Mercator"),
        ("projectionacronym", "merc"),
        ("ellipsoidacronym", "EPSG:7030"),
        ("geographicflag", "false"),
    ]
    for tag, text in values:
        ET.SubElement(spatialrefsys, tag).text = text


def add_color_ramp(renderer: ET.Element, colors: dict[str, tuple[str, str]]) -> None:
    shader = ET.SubElement(renderer, "rastershader")
    function = ET.SubElement(
        shader,
        "colorrampshader",
        {
            "colorRampType": "INTERPOLATED",
            "classificationMode": "1",
            "clip": "0",
            "labelPrecision": "6",
            "minimumValue": "0",
            "maximumValue": str(max(int(key) for key in colors)),
        },
    )
    for value, (label, color) in colors.items():
        ET.SubElement(
            function,
            "item",
            {"value": value, "label": label, "color": color, "alpha": color.split(",")[-1]},
        )


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def qgis_path(path: Path) -> str:
    """QGIS 工程内使用绝对路径，避免 qgz 解压到临时目录后图层丢失。"""
    return path.resolve().as_posix()


def add_layer(
    projectlayers: ET.Element,
    layer_id: str,
    output: dict[str, str | Path],
) -> None:
    path = Path(output["path"])
    with rasterio.open(path) as src:
        bounds = src.bounds

    maplayer = ET.SubElement(
        projectlayers,
        "maplayer",
        {
            "type": "raster",
            "styleCategories": "AllStyleCategories",
            "hasScaleBasedVisibilityFlag": "0",
            "readOnly": "0",
            "autoRefreshEnabled": "0",
            "autoRefreshMode": "Disabled",
            "autoRefreshTime": "0",
        },
    )
    extent = ET.SubElement(maplayer, "extent")
    ET.SubElement(extent, "xmin").text = str(bounds.left)
    ET.SubElement(extent, "ymin").text = str(bounds.bottom)
    ET.SubElement(extent, "xmax").text = str(bounds.right)
    ET.SubElement(extent, "ymax").text = str(bounds.top)
    ET.SubElement(maplayer, "id").text = layer_id
    ET.SubElement(maplayer, "datasource").text = qgis_path(path)
    ET.SubElement(maplayer, "keywordList")
    ET.SubElement(maplayer, "layername").text = str(output["name"])
    add_srs(maplayer)
    ET.SubElement(maplayer, "resourceMetadata")
    ET.SubElement(maplayer, "provider").text = "gdal"
    ET.SubElement(maplayer, "noData")
    style_manager = ET.SubElement(maplayer, "map-layer-style-manager", {"current": "default"})
    ET.SubElement(style_manager, "map-layer-style", {"name": "default"})
    pipe = ET.SubElement(maplayer, "pipe")
    provider = ET.SubElement(pipe, "provider")
    ET.SubElement(provider, "resampling", {"enabled": "false", "maxOversampling": "2"})
    renderer = ET.SubElement(
        pipe,
        "rasterrenderer",
        {"type": "singlebandpseudocolor", "opacity": "1", "alphaBand": "-1", "band": "1"},
    )
    if output["group"] == "classification":
        add_color_ramp(renderer, CLASS_COLORS)
    elif output["group"] == "vegetation":
        add_color_ramp(renderer, VEGETATION_COLORS)
    else:
        add_color_ramp(renderer, BINARY_COLORS)
    ET.SubElement(pipe, "brightnesscontrast", {"brightness": "0", "contrast": "0", "gamma": "1"})
    ET.SubElement(
        pipe,
        "huesaturation",
        {
            "colorizeOn": "0",
            "colorizeRed": "255",
            "colorizeGreen": "128",
            "colorizeBlue": "128",
            "colorizeStrength": "100",
            "grayscaleMode": "0",
            "saturation": "0",
        },
    )
    ET.SubElement(pipe, "rasterresampler", {"maxOversampling": "2"})
    ET.SubElement(maplayer, "blendMode").text = "0"


def build_project(outputs: list[dict[str, str | Path]]) -> tuple[Path, Path]:
    qgis = ET.Element("qgis", {"projectname": "U-Net六模型推理结果对比", "version": "3.34.0-Prizren"})
    ET.SubElement(qgis, "title").text = "U-Net六模型推理结果对比"
    tree_root = ET.SubElement(qgis, "layer-tree-group", {"name": "", "checked": "Qt::Checked", "expanded": "1"})
    canvas = ET.SubElement(qgis, "layer-tree-canvas")
    custom_order = ET.SubElement(canvas, "custom-order", {"enabled": "0"})
    projectlayers = ET.SubElement(qgis, "projectlayers")
    layer_ids: dict[Path, str] = {}

    for group_key, group_name in GROUPS.items():
        group = ET.SubElement(
            tree_root,
            "layer-tree-group",
            {"name": group_name, "checked": "Qt::Checked", "expanded": "1"},
        )
        for output in [item for item in outputs if item["group"] == group_key]:
            path = Path(output["path"])
            layer_id = f"{output['model']}_{group_key}_{uuid4().hex[:12]}"
            layer_ids[path] = layer_id
            ET.SubElement(
                group,
                "layer-tree-layer",
                {
                    "id": layer_id,
                    "name": str(output["name"]),
                    "checked": "Qt::Checked",
                    "expanded": "1",
                    "providerKey": "gdal",
                    "source": qgis_path(path),
                },
            )
            ET.SubElement(custom_order, "item").text = layer_id
            add_layer(projectlayers, layer_id, output)

    layerorder = ET.SubElement(qgis, "layerorder")
    for output in outputs:
        ET.SubElement(layerorder, "layer", {"id": layer_ids[Path(output["path"])]})

    properties = ET.SubElement(qgis, "properties")
    srs = ET.SubElement(properties, "SpatialRefSys")
    ET.SubElement(srs, "ProjectCrs", {"type": "QString"}).text = "EPSG:3857"
    ET.SubElement(srs, "ProjectCRSID", {"type": "int"}).text = "3857"
    ET.SubElement(srs, "ProjectionsEnabled", {"type": "int"}).text = "1"

    qgs_path = PROJECT_ROOT / "qilian_unet_model_compare_absolute.qgs"
    qgz_path = PROJECT_ROOT / "qilian_unet_model_compare_absolute.qgz"
    pretty = minidom.parseString(ET.tostring(qgis, encoding="utf-8")).toprettyxml(
        indent="  ", encoding="utf-8"
    )
    qgs_path.write_bytes(pretty)
    with zipfile.ZipFile(qgz_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(qgs_path, arcname=qgs_path.name)
    return qgs_path, qgz_path


def main() -> None:
    outputs = copy_named_outputs()
    qgs_path, qgz_path = build_project(outputs)
    print(f"整理目录：{PROJECT_ROOT}")
    print(f"QGIS 工程：{qgz_path}")
    print(f"QGS 备份：{qgs_path}")
    print(f"分类/建筑/植被 GeoTIFF：{len(outputs)} 个")


if __name__ == "__main__":
    main()
