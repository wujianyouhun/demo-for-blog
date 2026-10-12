"""将六模型的分类、建筑、植被结果制作成带模型名和图例的 1 秒 GIF。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "out" / "qilian_multiclass_5class_refined" / "qgis_compare"
OUTPUT = SOURCE / "gifs"
MODELS = (
    ("unet", "U-Net"),
    ("resunet", "ResUNet"),
    ("attention_unet", "Attention U-Net"),
    ("unetpp", "U-Net++"),
    ("transunet", "TransUNet"),
    ("swin_unet", "Swin-UNet"),
)

# 所有 GIF 共用同一套地物颜色，与 GeoTIFF/QGIS 对比工程保持一致。
COLORS = {
    "background": (247, 248, 250),
    "text": (31, 41, 55),
    "muted": (107, 114, 128),
    "road": (220, 50, 47),
    "building": (255, 165, 0),
    "vegetation": (76, 175, 80),
    "bare_land": (156, 112, 69),
    "solar": (33, 150, 243),
}
CLASS_COLORS = np.array(
    [COLORS["road"], COLORS["building"], COLORS["vegetation"], COLORS["bare_land"], COLORS["solar"]],
    dtype=np.uint8,
)
LONG_EDGE = 1280
HEADER_HEIGHT = 66
FOOTER_HEIGHT = 72


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先使用 Windows 中文字体；缺失时仍可生成带英文模型名的 GIF。"""
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf") if bold else Path("C:/Windows/Fonts/simsun.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def output_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, LONG_EDGE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def read_labels(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        target_width, target_height = output_size(source.width, source.height)
        return source.read(
            1,
            out_shape=(1, target_height, target_width),
            resampling=rasterio.enums.Resampling.nearest,
        )


def render_labels(labels: np.ndarray, kind: str) -> Image.Image:
    rgb = np.full((*labels.shape, 3), COLORS["background"], dtype=np.uint8)
    if kind == "classification":
        valid = (labels >= 0) & (labels < len(CLASS_COLORS))
        rgb[valid] = CLASS_COLORS[labels[valid]]
    elif kind == "building":
        rgb[labels == 1] = COLORS["building"]
    else:
        rgb[labels == 1] = COLORS["vegetation"]
    return Image.fromarray(rgb, mode="RGB")


def draw_legend(draw: ImageDraw.ImageDraw, kind: str, top: int, image_width: int) -> None:
    entries = {
        "classification": (
            ("道路", COLORS["road"]),
            ("居民地/建筑", COLORS["building"]),
            ("耕地/植被", COLORS["vegetation"]),
            ("裸地", COLORS["bare_land"]),
            ("光伏场", COLORS["solar"]),
        ),
        "building": (("建筑 / 居民地", COLORS["building"]),),
        "vegetation": (("植被 / 耕地", COLORS["vegetation"]),),
    }[kind]
    legend_font = font(22)
    x = 28
    for label, color in entries:
        draw.rounded_rectangle((x, top + 18, x + 25, top + 43), radius=4, fill=color)
        x += 35
        draw.text((x, top + 15), label, font=legend_font, fill=COLORS["text"])
        text_box = draw.textbbox((x, top + 15), label, font=legend_font)
        x = text_box[2] + 34
    draw.text((image_width - 184, top + 17), "1 秒 / 帧 · 循环", font=font(18), fill=COLORS["muted"])


def make_frame(labels: np.ndarray, kind: str, model_name: str, index: int) -> Image.Image:
    map_image = render_labels(labels, kind)
    width, height = map_image.size
    frame = Image.new("RGB", (width, height + HEADER_HEIGHT + FOOTER_HEIGHT), "white")
    frame.paste(map_image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(frame)
    titles = {"classification": "五分类结果对比", "building": "建筑 / 居民地提取对比", "vegetation": "植被 / 耕地提取对比"}
    draw.text((28, 16), titles[kind], font=font(28, bold=True), fill=COLORS["text"])
    draw.text((width - 215, 21), f"{model_name}  ·  {index}/6", font=font(20), fill=COLORS["muted"])
    draw.line((0, HEADER_HEIGHT - 1, width, HEADER_HEIGHT - 1), fill=(226, 232, 240), width=1)
    draw.line((0, HEADER_HEIGHT + height, width, HEADER_HEIGHT + height), fill=(226, 232, 240), width=1)
    draw_legend(draw, kind, HEADER_HEIGHT + height, width)
    return frame


def gif_paths(kind: str) -> tuple[Path, str]:
    names = {
        "classification": ("classification_6_models_1s.gif", "classification"),
        "building": ("building_6_models_1s.gif", "building"),
        "vegetation": ("vegetation_6_models_1s.gif", "vegetation"),
    }
    return OUTPUT / names[kind][0], names[kind][1]


def create_gif(kind: str) -> Path:
    output_path, filename_suffix = gif_paths(kind)
    frames = []
    for index, (model_id, model_name) in enumerate(MODELS, start=1):
        raster_path = SOURCE / kind / f"{model_id}_{filename_suffix}.tif"
        if not raster_path.is_file():
            raise FileNotFoundError(f"缺少 {model_name} 的 {kind} 结果：{raster_path}")
        frames.append(make_frame(read_labels(raster_path), kind, model_name, index))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return output_path


def main() -> None:
    for kind in ("classification", "building", "vegetation"):
        path = create_gif(kind)
        print(f"已生成：{path}")


if __name__ == "__main__":
    main()
