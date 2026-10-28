"""只读核查实际样本与已有模型历史，生成诊断 JSON，不推理整幅影像。"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from common import ROOT, MODEL_DIR
from optimization import dataset_counts, metrics_from_confusion
from train_remote_sensing import TileDataset


def audit(dataset, checkpoint):
    meta=torch.load(checkpoint,map_location="cpu",weights_only=True)
    report=dict(checkpoint=str(checkpoint.resolve()),dataset=str(dataset.resolve()),
                model={k:meta.get(k) for k in ["architecture","class_names","bands","scale","epoch","val_miou","normalization","in_channels"]},
                splits={},unet_experiments=[])
    names=meta["class_names"]
    config=dataset/"dataset.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")).get("class_names")!=list(names):
        raise ValueError("数据集与权重类别不一致。")
    for split in ["train","val","test"]:
        if not list((dataset/split/"images").glob("*.tif")):continue
        ds=TileDataset(dataset,split,meta["bands"],meta["scale"],meta["tile_size"],len(names),meta.get("normalization"))
        rows=dataset_counts(ds)
        counts=rows.sum(0)
        record=dict(tiles=len(ds),pixels=counts.tolist(),fractions=(counts/counts.sum()).tolist(),
                    tiles_containing_class=(rows>0).sum(0).tolist())
        with rasterio.open(ds.items[0][0]) as src:
            record["example"]=dict(crs=str(src.crs),resolution=list(src.res),bands=src.count,dtype=src.dtypes[0])
        report["splits"][split]=record
    # 仅训练像元抽样用于输入诊断，不从验证集拟合任何预处理。
    samples=[]
    ds=TileDataset(dataset,"train",meta["bands"],meta["scale"],meta["tile_size"],len(names),meta.get("normalization"))
    for image,mask in ds.items:
        with rasterio.open(image) as src,rasterio.open(mask) as labels:
            raw=src.read(meta["bands"])
            valid=(src.read_masks(meta["bands"])>0).all(0)&np.isfinite(raw).all(0)
            valid&=(labels.read(1)!=255)&(labels.read_masks(1)>0)
            samples.append(raw[:,valid][:,::16])
    raw=np.concatenate(samples,axis=1)
    report["train_input_sample"]=dict(stride=16,percentiles_0_2_50_98_100=np.percentile(raw,[0,2,50,98,100],axis=1).T.tolist(),
                                      below_zero=(raw<0).mean(1).tolist(),above_scale=(raw>meta["scale"]).mean(1).tolist())
    if meta.get("validation",{}).get("confusion_matrix"):
        report["saved_validation"]=metrics_from_confusion(meta["validation"]["confusion_matrix"],names)
    for path in (ROOT.parent/"models/Unet").rglob("training_summary.json"):
        for experiment in json.loads(path.read_text(encoding="utf-8")):
            best=max(experiment["history"],key=lambda h:h["miou"])
            report["unet_experiments"].append(dict(source=str(path.resolve()),model=experiment["model"],
                                                  epochs=len(experiment["history"]),best_epoch=best["epoch"],
                                                  miou=experiment["best_miou"]))
    report["interpretation"]=["U-Net 与 DeepLabV3 类别、样本、通道、归一化和训练预算不同，不能直接排序。",
                               "输入波段顺序/缩放须核实，百分位不能证明波段含义。",
                               "相邻验证条带不代表独立区域测试。"]
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir",type=Path,default=ROOT/"data/landcover_6class")
    p.add_argument("--checkpoint",type=Path,default=MODEL_DIR/"landcover_6class_best.pth")
    p.add_argument("--output",type=Path,default=ROOT/"out/optimization/baseline_audit.json")
    a=p.parse_args()
    result=audit(a.dataset_dir,a.checkpoint)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"核查报告：{a.output.resolve()}")
    print("无预测像元的类别：",result.get("saved_validation",{}).get("zero_prediction_classes"))


if __name__=="__main__":main()
