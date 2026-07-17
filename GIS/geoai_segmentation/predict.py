from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image

from train import build_model
from utils import mask_to_vectors, read_image


def predict(input_path: Path, checkpoint_path: Path, output_dir: Path, threshold=.5):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state=torch.load(checkpoint_path,map_location=device,weights_only=False)
    model=build_model(state["model"],state.get("base_channels",32)).to(device)
    model.load_state_dict(state["state_dict"]); model.eval()
    image=read_image(input_path); height,width=image.shape[:2]; size=int(state.get("size",256))
    resized=cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
    tensor=torch.from_numpy(np.moveaxis(resized.astype(np.float32)/255,-1,0)).unsqueeze(0).to(device)
    with torch.no_grad(): mask=model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
    mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
    output_dir.mkdir(parents=True,exist_ok=True); preview=output_dir/"prediction.png"; Image.fromarray(mask*255).save(preview)
    result={"preview":str(preview)}
    if input_path.suffix.lower() in {".tif",".tiff"}:
        with rasterio.open(input_path) as source: profile=source.profile.copy(); transform=source.transform; crs=source.crs
        profile.update(count=1,dtype="uint8",nodata=255,compress="deflate")
        mask_path=output_dir/"prediction.tif"
        with rasterio.open(mask_path,"w",**profile) as sink: sink.write(mask,1)
        vectors=mask_to_vectors(mask,transform,crs,output_dir/"prediction.gpkg")
        result.update({"mask":str(mask_path),"vectors":vectors})
    return result


def parser():
    result=argparse.ArgumentParser(description="语义分割基线推理")
    result.add_argument("--input",required=True); result.add_argument("--checkpoint",required=True); result.add_argument("--output",default="outputs/prediction")
    return result


if __name__=="__main__":
    args=parser().parse_args(); print(json.dumps(predict(Path(args.input),Path(args.checkpoint),Path(args.output)),ensure_ascii=False,indent=2))
