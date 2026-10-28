"""用真实评价 JSON 与同配置推理结果生成博客图，不重新训练。"""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'assets' / 'optimization'
OUT.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/msyh.ttc'
LABELS = ['道路', '建设用地', '耕地', '裸地', '水体', '光伏']
COLORS = [(230,65,60),(245,165,40),(65,180,75),(185,155,110),(45,120,235),(185,70,210)]

def txt(draw, pos, value, size=22, color='#334155'):
    draw.text(pos, str(value), font=ImageFont.truetype(FONT,size), fill=color)

def canvas(title, subtitle, width=1200, height=720):
    im=Image.new('RGB',(width,height),'#f7fafc');d=ImageDraw.Draw(im)
    txt(d,(40,24),title,32,'#123d4b');txt(d,(40,80),subtitle,20,'#64748b')
    return im,d

def read(path):return json.loads(path.read_text(encoding='utf-8'))
baseline=read(ROOT/'out/optimization/baseline_evaluation.json')
trial=read(ROOT/'out/optimization/trial_evaluation.json')
assert baseline['dataset_sha256']==trial['dataset_sha256']
audit=read(ROOT/'out/optimization/baseline_audit.json')
history=read(ROOT.parent/'models/deeplabv3/landcover_6class_optimized_trial/history.json')

im,d=canvas('样本失衡：耕地与裸地占训练像元的 93.25%', '真实训练像元统计；道路、水体、光伏各自均不足 1%。')
counts=audit['splits']['train']['pixels'];total=sum(counts)
for i,(name,count,color) in enumerate(zip(LABELS,counts,COLORS)):
    y=145+i*72
    txt(d,(42,y),name,24)
    d.rectangle((200,y+5,1010,y+37),fill='#e7eef2')
    d.rectangle((200,y+5,200+810*count/total,y+37),fill=color)
    txt(d,(1025,y),f'{count/total:.2%}',23)
    txt(d,(200,y+42),f'{count:,} 像元',17,'#64748b')
txt(d,(42,630),'横轴满格为 100%；统计不包含 255 忽略像元。',21)
im.save(OUT/'01_class_balance.png')

im,d=canvas('六轮候选试验：分数波动，最佳值接近原基线', '蓝绿色实线：候选每轮验证 mIoU；灰色虚线：原权重 0.215706 参考线。')
x0,y0,w,h=110,560,990,390
for i in range(7):
    v=i*.05;y=y0-v/.3*h
    d.line((x0,y,x0+w,y),fill='#dce5eb',width=1);txt(d,(42,y-13),f'{v:.2f}',18)
by=y0-baseline['metrics']['miou']/.3*h
for x in range(x0,x0+w,20):d.line((x,by,x+11,by),fill='#64748b',width=2)
points=[]
for i,row in enumerate(history):
    x=x0+40+i*180;y=y0-row['validation']['miou']/.3*h
    points.append((x,y))
    label_y=y+14 if y>by+5 else y-36
    txt(d,(x-35,label_y),f"{row['validation']['miou']:.4f}",19)
    txt(d,(x-5,y0+15),row['epoch'],20)
d.line(points,fill='#148a8a',width=4)
for x,y in points:d.ellipse((x-6,y-6,x+6,y+6),fill='#148a8a')
txt(d,(40,635),'最佳第 4 轮：0.215951；较原基线仅增加 0.000245，尚无实质改善。',23,'#9a4b35')
im.save(OUT/'02_training_curve.png')

im,d=canvas('逐类 IoU：平均分的小幅变化没有解决缺失类别', '灰色：原模型；蓝绿色：优化候选。两者使用完全相同的验证切片。',height=820)
for i,name in enumerate(LABELS):
    y=150+i*88;txt(d,(35,y),name,23)
    old=baseline['metrics']['per_class_iou'][i];new=trial['metrics']['per_class_iou'][i]
    for j,(v,c) in enumerate([(old,'#94a3b8'),(new,'#148a8a')]):
        yy=y+j*30;d.rectangle((200,yy+2,850,yy+22),fill='#e7eef2')
        if v>0:d.rectangle((200,yy+2,200+650*v,yy+22),fill=c)
        txt(d,(875,yy-4),f'{v:.6f}',21)
txt(d,(40,720),'横轴满格为 IoU=1。建设用地候选 IoU=0.000318，仍几乎没有有效识别。',21)
im.save(OUT/'03_class_iou.png')

im,d=canvas('混淆矩阵：真实少数类主要被判成耕地', '每行按真实类别归一化为百分比；纵轴是真值，横轴是预测。',width=1600,height=880)
for panel,(title,result) in enumerate([('原模型',baseline),('候选最佳模型',trial)]):
    ox=145+panel*790;oy=220;cell=91
    txt(d,(ox,130),title,28,'#126d75')
    matrix=np.asarray(result['metrics']['confusion_matrix'],dtype=float)
    matrix=matrix/np.maximum(matrix.sum(1,keepdims=True),1)
    for i,name in enumerate(LABELS):
        txt(d,(ox+i*cell+4,oy-45),name,18)
        txt(d,(ox-119,oy+i*cell+31),name,19)
        for j in range(6):
            v=matrix[i,j];start=np.array([239,247,248]);end=np.array([20,112,122]);c=tuple((start*(1-v)+end*v).astype(int))
            d.rectangle((ox+j*cell,oy+i*cell,ox+(j+1)*cell-2,oy+(i+1)*cell-2),fill=c)
            txt(d,(ox+j*cell+12,oy+i*cell+31),f'{v:.1%}',18,'white' if v>.55 else '#334155')
txt(d,(40,807),'候选建设用地真阳性仅 12 像元；道路、水体、光伏仍没有预测。',23,'#9a4b35')
im.save(OUT/'04_confusion_matrices.png')

# 使用同一显示拉伸、相同颜色、相同透明度生成四格对照。
with rasterio.open(ROOT/'data/train/image2.tif') as src:
    rgb=src.read([1,2,3]).astype('float32');profile=(src.shape,src.crs,src.transform)
for channel in rgb:
    lo,hi=np.percentile(channel,[2,98]);channel[:]=np.clip((channel-lo)/max(hi-lo,1),0,1)
rgb=(rgb.transpose(1,2,0)*255).astype('uint8')
def load_mask(path):
    with rasterio.open(path) as src:
        assert (src.shape,src.crs,src.transform)==profile
        return src.read(1)
truth=load_mask(ROOT/'data/landcover_6class/full_masks/image2_mask.tif')
old=load_mask(ROOT/'out/blog_optimization/baseline_image2_center.tif')
new=load_mask(ROOT/'out/blog_optimization/trial_image2_center.tif')
def overlay(mask):
    valid=mask!=255;result=rgb.copy()
    result[valid]=(rgb[valid]*.55+np.array(COLORS,dtype='uint8')[mask[valid]]*.45).astype('uint8')
    return result
pw,ph=728,581
im,d=canvas('同一影像、同一推理方式：原模型与候选权重对照', 'image2 含训练区域，仅用于观察表现；不是独立测试。显示统一为 RGB 2%–98% 拉伸。',width=1532,height=1510)
for i,(label,array) in enumerate([('影像 RGB',rgb),('人工标注（不是模型预测）',overlay(truth)),
                                ('原权重：256 窗口 / halo=32',overlay(old)),('候选权重：256 窗口 / halo=32',overlay(new))]):
    x=25+i%2*(pw+26);y=175+i//2*(ph+65)
    txt(d,(x,y-40),label,23,'#126d75');im.paste(Image.fromarray(array),(x,y))
for i,(label,color) in enumerate(zip(LABELS,COLORS)):
    x=30+i*245;y=1435;d.rectangle((x,y,x+20,y+20),fill=color);txt(d,(x+28,y-4),label,20)
im.save(OUT/'05_image2_controlled_comparison.jpg',quality=94)

sources=[ROOT/'out/optimization/baseline_evaluation.json',ROOT/'out/optimization/trial_evaluation.json',
         ROOT/'out/optimization/baseline_audit.json',ROOT.parent/'models/deeplabv3/landcover_6class_optimized_trial/history.json']
manifest=[]
for p in sources:
    target=OUT/('trial_history.json' if p.name=='history.json' else p.name)
    shutil.copy2(p,target)
    manifest.append(dict(file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(OUT/'figure_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)
