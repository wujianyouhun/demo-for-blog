"""优化流程的数值、同步增强、地理网格与精确断点恢复回归检查。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from common import to_tensor, normalization_for_channels, training_model
from optimization import (segmentation_loss, sampling_weights, paired_transform,
                          metrics_from_confusion)
from infer_optimized import predict, probabilities
from evaluate_raster import evaluate
import train_optimized


class TinyModel(torch.nn.Module):
    def __init__(self, classes=3, channels=3):
        super().__init__()
        self.backbone=torch.nn.Conv2d(channels,4,1)
        self.classifier=torch.nn.Conv2d(4,classes,1)

    def forward(self,x):
        return {"out":self.classifier(self.backbone(x).relu())}


def write(path,array,nodata=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if array.ndim==2:array=array[None]
    with rasterio.open(path,"w",driver="GTiff",height=array.shape[-2],width=array.shape[-1],
                       count=array.shape[0],dtype=array.dtype,crs="EPSG:32647",
                       transform=from_origin(500000,4200000,10,10),nodata=nodata) as dst:dst.write(array)


class OptimizationTests(unittest.TestCase):
    def test_ignore_and_dice_gradients(self):
        logits=torch.randn(1,3,3,4,requires_grad=True)
        target=torch.tensor([[[0,1,255,2],[0,1,2,255],[0,1,2,255]]])
        weights=torch.tensor([1.,2.,3.])
        baseline=segmentation_loss(logits,target,weights,.3)
        changed=logits.detach().clone()
        changed.permute(0,2,3,1)[target==255]=1000
        self.assertAlmostEqual(float(baseline.detach()),float(segmentation_loss(changed,target,weights,.3)),places=6)
        baseline.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.permute(0,2,3,1)[target==255]==0).all())
        all_ignore=torch.full_like(target,255)
        self.assertEqual(float(segmentation_loss(logits,all_ignore,weights,.3).detach()),0.)

    def test_pairing_sampling_and_metrics(self):
        mask=torch.arange(16).reshape(4,4)
        image=mask.repeat(3,1,1).float()
        x,y=paired_transform(image,mask,True,True,3)
        torch.testing.assert_close(x[0],y.float())
        w=sampling_weights(torch.tensor([[100,0],[0,1],[50,1]]),3)
        self.assertGreater(float(w[1]),float(w[0]))
        self.assertLessEqual(float(w.max()/w.min()),3.)
        m=metrics_from_confusion([[0,3],[0,2]],["a","b"])
        self.assertEqual(m["zero_prediction_classes"],["a"])
        self.assertAlmostEqual(m["miou"],.2)
        self.assertEqual(m["per_class"][0]["precision"],None)

    def test_four_band_normalization_and_model(self):
        x=to_tensor(np.ones((4,32,32)),10000.,normalization_for_channels(4))
        self.assertEqual(tuple(x.shape),(4,32,32))
        with self.assertRaises(ValueError):to_tensor(np.ones((4,2,2)),10000.,normalization_for_channels(3))
        model=training_model(6,pretrained=False,in_channels=4,aspp_rates=[6,12,18]).eval()
        self.assertEqual(model.classifier[0].convs[3][0].dilation,(18,18))
        with torch.no_grad():self.assertEqual(tuple(model(x[None])["out"].shape),(1,6,32,32))

    def test_probability_fusion_and_nodata(self):
        torch.manual_seed(8)
        model=TinyModel().eval()
        value=torch.randn(1,3,32,32)
        # 逐点网络具翻转等变性，逆变换 TTA 必须回到相同位置。
        np.testing.assert_allclose(probabilities(model,value),probabilities(model,value,True),atol=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for h,w in [(5,7),(45,51)]:
                image=np.full((3,h,w),100,dtype="uint8");image[:,0,0]=0
                source=root/f"source{h}.tif";output=root/f"output{h}.tif";confidence=root/f"confidence{h}.tif"
                write(source,image)
                meta=dict(bands=[1,2,3],scale=255.,tile_size=32,class_names=["a","b","c"])
                with patch('infer_optimized.load_checkpoint',return_value=(model,meta)):
                    result=predict(source,root/'unused.pth',output,'cpu',32,16,True,True,confidence)
                self.assertEqual(result["valid_pixels"],h*w-1)
                with rasterio.open(source) as src,rasterio.open(output) as pred,rasterio.open(confidence) as conf:
                    labels=pred.read(1);scores=conf.read(1)
                    self.assertEqual(src.transform,pred.transform);self.assertEqual(src.crs,pred.crs)
                    self.assertEqual(labels[0,0],255);self.assertEqual(scores[0,0],-1.)
                    self.assertTrue(((scores[labels!=255]>=0)&(scores[labels!=255]<=1.00001)).all())
                    self.assertEqual(labels[-1,-1]==255,False)
                truth=root/f'truth{h}.tif';write(truth,labels,255)
                self.assertEqual(evaluate(output,truth,meta["class_names"])["accuracy"],1.)
                invalid=labels.copy();invalid[-1,-1]=255
                missing=root/f'missing{h}.tif';write(missing,invalid,255)
                with self.assertRaisesRegex(ValueError,'缺失或无效预测'):
                    evaluate(missing,truth,meta['class_names'])
                bad_truth=root/f'bad_truth{h}.tif';write(bad_truth,labels,0)
                with self.assertRaisesRegex(ValueError,'NoData'):
                    evaluate(output,bad_truth,meta['class_names'])

    def test_exact_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dataset=root/'dataset';dataset.mkdir()
            (dataset/'dataset.json').write_text(json.dumps(dict(class_names=['a','b','c'],bands=[1,2,3],scale=255.,tile_size=32)))
            for split,n in [('train',3),('val',1)]:
                for i in range(n):
                    rng=np.random.default_rng(i)
                    image=rng.integers(0,256,(3,32,32),dtype='uint8')
                    labels=(np.indices((32,32))[1]%3).astype('uint8');labels[0]=255
                    write(dataset/split/'images'/f'{i}.tif',image)
                    write(dataset/split/'masks'/f'{i}_mask.tif',labels,255)
            common=['--dataset-dir',str(dataset),'--epochs','2','--head-only-epochs','0',
                    '--batch-size','2','--threads','1','--patience','0','--device','cpu']
            factory=lambda classes,pretrained,in_channels,aspp_rates:TinyModel(classes,in_channels)
            with patch('train_optimized.training_model',side_effect=factory):
                train_optimized.main(common+['--output-dir',str(root/'full'),'--from-scratch'])
                original_save=train_optimized.atomic_save
                def interrupt_after_last(value,path):
                    original_save(value,path)
                    if path.name=='last.pth':raise InterruptedError('simulate process interruption')
                with patch('train_optimized.atomic_save',side_effect=interrupt_after_last):
                    with self.assertRaises(InterruptedError):
                        train_optimized.main(common+['--output-dir',str(root/'resumed'),'--from-scratch'])
                train_optimized.main(common+['--output-dir',str(root/'resumed'),'--resume',str(root/'resumed/last.pth')])
            full=torch.load(root/'full/last.pth',weights_only=True)
            resumed=torch.load(root/'resumed/last.pth',weights_only=True)
            for key,value in full['model'].items():torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
            self.assertEqual(full['history'][-1]['validation'],resumed['history'][-1]['validation'])
            self.assertEqual(full['scheduler'],resumed['scheduler'])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
