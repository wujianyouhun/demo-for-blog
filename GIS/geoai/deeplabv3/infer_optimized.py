"""重叠窗口概率融合和可选翻转 TTA，磁盘暂存大图概率，保留原始地理网格。"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from common import ROOT, IGNORE, choose_device, load_checkpoint, to_tensor, palette


def spatial_weights(size):
    # 正权重保证边界和单窗口影像也有定义；中心占比较大。
    axis = np.maximum(np.hanning(size).astype("float32"), .05)
    return np.outer(axis, axis)


@torch.inference_mode()
def probabilities(model, image, tta=False):
    result = None
    flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    for dims in flips:
        logits = model(image.flip(dims) if dims else image)["out"]
        probability = logits.softmax(1)
        if dims:
            probability = probability.flip(dims)
        result = probability if result is None else result + probability
    return (result / len(flips))[0].cpu().numpy()


def predict(image_path, checkpoint_path, output_path, device_name="auto", tile_size=None,
            stride=None, tta=False, zero_is_nodata=False, confidence_path=None, scratch_dir=None):
    device = choose_device(device_name)
    model, meta = load_checkpoint(checkpoint_path, device)
    size = tile_size or meta.get("tile_size", 256)
    stride = stride if stride is not None else size // 2
    if size < 32 or not 0 < stride <= size:
        raise ValueError("tile-size 至少为 32，stride 位于 1..tile-size。")
    outputs = [output_path] + ([confidence_path] if confidence_path else [])
    paths = [image_path, checkpoint_path, *outputs]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("输入、权重、分类和概率文件路径不得相同。")
    if any(p.exists() for p in outputs):
        raise FileExistsError("输出文件已存在，请指定新路径。")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    bands, names = meta["bands"], meta["class_names"]
    counts = np.zeros(len(names), dtype="int64")
    weight = spatial_weights(size)
    with rasterio.open(image_path) as src, tempfile.TemporaryDirectory(dir=scratch_dir) as temporary:
        if max(bands) > src.count or src.crs is None:
            raise ValueError("输入缺少所需波段或 CRS。")
        shape = (src.height, src.width)
        sums = np.memmap(Path(temporary)/"probabilities.bin", mode="w+", dtype="float32", shape=(len(names), *shape))
        weights = np.memmap(Path(temporary)/"weights.bin", mode="w+", dtype="float32", shape=shape)
        sums[:] = 0; weights[:] = 0
        try:
            for y in range(0, src.height, stride):
                for x in range(0, src.width, stride):
                    h, w = min(size, src.height-y), min(size, src.width-x)
                    window = Window(x, y, w, h)
                    image = src.read(bands, window=window)
                    valid = (src.read_masks(bands, window=window) > 0).all(0) & np.isfinite(image).all(0)
                    if zero_is_nodata:
                        valid &= ~np.all(image == 0, axis=0)
                    if not valid.any():
                        continue
                    # 复制边缘只提供网络上下文，不将补齐像元作为有效输出。
                    image = np.pad(image, ((0,0),(0,size-h),(0,size-w)), mode="edge")
                    tensor = to_tensor(image, meta["scale"], meta.get("normalization")).unsqueeze(0).to(device)
                    probs = probabilities(model, tensor, tta)[:, :h, :w]
                    local = weight[:h, :w] * valid
                    sums[:, y:y+h, x:x+w] += probs * local
                    weights[y:y+h, x:x+w] += local
                print(f"概率融合 {min(y+stride,src.height)}/{src.height} 行", flush=True)
            profile = dict(driver="GTiff", height=src.height, width=src.width, count=1,
                           crs=src.crs, transform=src.transform, dtype="uint8", nodata=IGNORE,
                           compress="deflate", BIGTIFF="IF_SAFER")
            conf = None
            try:
                if confidence_path:
                    conf = rasterio.open(confidence_path, "w", **(profile | dict(dtype="float32", nodata=-1.)))
                with rasterio.open(output_path,"w",**profile) as dst:
                    for y in range(0, src.height, 256):
                        for x in range(0, src.width, 256):
                            h, w = min(256,src.height-y), min(256,src.width-x)
                            ws = weights[y:y+h,x:x+w]
                            valid = ws > 0
                            probability = sums[:,y:y+h,x:x+w] / np.maximum(ws,1e-20)
                            labels = probability.argmax(0).astype("uint8")
                            labels[~valid] = IGNORE
                            win = Window(x,y,w,h)
                            dst.write(labels,1,window=win)
                            counts += np.bincount(labels[valid],minlength=len(names))
                            if conf:
                                maximum = probability.max(0).astype("float32")
                                maximum[~valid] = -1
                                conf.write(maximum,1,window=win)
                    colors = palette(256)
                    named = dict(road=(230,65,60),built_up=(245,165,40),cropland=(65,180,75),
                                 bare_land=(185,155,110),water=(45,120,235),solar_farm=(185,70,210))
                    cmap = {i: tuple(named.get(name, colors[3*i:3*i+3]))+(255,) for i,name in enumerate(names)}
                    dst.write_colormap(1,cmap | {255:(0,0,0,0)})
                    tags = dict(class_names=json.dumps(names), checkpoint=str(checkpoint_path.resolve()),
                                class_labels=json.dumps(meta.get("class_labels") or names,ensure_ascii=False),
                                bands=json.dumps(bands), scale=str(meta["scale"]),
                                normalization=json.dumps(meta.get("normalization")),
                                tile_size=str(size),stride=str(stride),tta=str(tta),zero_is_nodata=str(zero_is_nodata),
                                fusion="positive_hann_probability_average",checkpoint_epoch=str(meta.get("epoch","")))
                    dst.update_tags(**tags)
                    if conf:
                        conf.update_tags(**tags,description="maximum softmax probability; uncalibrated")
            finally:
                if conf:
                    conf.close()
        finally:
            # Windows 上必须先关闭映射才能清理临时目录。
            sums._mmap.close(); weights._mmap.close()
    total = int(counts.sum())
    summary = dict(image=str(image_path.resolve()),checkpoint=str(checkpoint_path.resolve()),
                   tile_size=size,stride=stride,tta=tta,zero_is_nodata=zero_is_nodata,
                   classes=[dict(class_id=i,name=name,pixels=int(counts[i]),fraction=float(counts[i]/total) if total else 0.)
                            for i,name in enumerate(names)],valid_pixels=total)
    output_path.with_suffix(".classes.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,default=ROOT/"out/optimized/classification.tif")
    p.add_argument("--confidence",type=Path)
    p.add_argument("--tile-size",type=int)
    p.add_argument("--stride",type=int)
    p.add_argument("--tta",action="store_true")
    p.add_argument("--zero-is-nodata",action="store_true",help="仅在确认所选波段全零代表填充时使用")
    p.add_argument("--scratch-dir",type=Path,help="概率磁盘映射暂存目录，需要足够空间")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--device",default="auto")
    a=p.parse_args()
    if a.threads<1:p.error("threads 必须为正。")
    torch.set_num_threads(a.threads)
    result=predict(a.image,a.checkpoint,a.output,a.device,a.tile_size,a.stride,a.tta,
                   a.zero_is_nodata,a.confidence,a.scratch_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
