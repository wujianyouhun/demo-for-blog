"""评价融合/TTA 后的分类栅格：必须与单波段真值严格同网格，不自动重采样。"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from common import IGNORE
from optimization import add_confusion, metrics_from_confusion


def evaluate(prediction, truth, names):
    matrix=torch.zeros((len(names),len(names)),dtype=torch.int64)
    with rasterio.open(prediction) as pred,rasterio.open(truth) as target:
        if (pred.count!=1 or target.count!=1 or pred.shape!=target.shape or pred.crs!=target.crs
                or not pred.transform.almost_equals(target.transform)):
            raise ValueError("预测与真值必须为单波段、同尺寸、同 CRS 和同仿射变换。")
        if target.nodata is not None and 0 <= target.nodata < len(names):
            raise ValueError("真值 NoData 不得使用有效类别编号。")
        tags=pred.tags()
        if "class_names" in tags and json.loads(tags["class_names"])!=list(names):
            raise ValueError("预测类别名称/顺序与显式类别表不一致。")
        for _,window in target.block_windows(1):
            raw=target.read(1,window=window)
            valid=(target.read_masks(1,window=window)>0)&(raw!=IGNORE)
            if np.any(valid & (~np.isfinite(raw) | (raw!=np.floor(raw)))):
                raise ValueError("有效真值必须为有限整数类别。")
            y=np.where(valid,raw,IGNORE).astype("int64")
            yp=pred.read(1,window=window)
            pv=pred.read_masks(1,window=window)>0
            if np.any(valid & (~pv | (yp<0) | (yp>=len(names)) | (yp!=np.floor(yp)))):
                raise ValueError("真值有效区域存在缺失或无效预测，不能静默排除后计算精度。")
            if np.any(valid & ((y<0)|(y>=len(names)))):
                raise ValueError("真值类别越界。")
            y[~valid]=IGNORE
            add_confusion(matrix,torch.from_numpy(yp.astype("int64")),torch.from_numpy(y))
    result=metrics_from_confusion(matrix,names)
    if not result["valid_pixels"]:raise ValueError("没有有效真值像元。")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction",type=Path,required=True)
    p.add_argument("--truth",type=Path,required=True)
    p.add_argument("--class-names",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.prediction,a.truth,a.class_names)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
