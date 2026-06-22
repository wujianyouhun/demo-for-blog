# 快速上手

## 1. 安装环境

```bash
conda activate geoai
python -m pip install -r requirements.txt
```

项目已按 `geoai-py==0.40.0` 验证。也可以新建隔离环境：

```bash
conda env create -f environment.yml
conda activate change-detection
```

首次运行 GeoAI ChangeStar 可能会下载预训练模型，需要联网。模型缓存目录由 `config.py` 设置到 `data/pretrained`。

## 2. GeoAI 下载双时相数据

```bash
python scripts/download_data.py --region beijing --date-a 2022-06-01 --date-b 2023-06-01 --cloud 20
```

下载结果位于：

```text
data/raw/time_a/
data/raw/time_b/
```

## 3. 直接执行 GeoAI 变化检测

GeoAI 模式不需要 `--model` 参数。

```bash
python scripts/predict.py \
  --engine geoai \
  --image-a data/raw/time_a/xxx_A.tif \
  --image-b data/raw/time_b/xxx_B_aligned.tif \
  --visualize
```

输出位于 `data/output/`，包括变化掩膜、变化矢量、T1/T2 建筑语义图和叠加预览图。

## 4. 制作训练样本

模拟样本，适合快速学习训练流程：

```bash
python scripts/generate_sample.py --mode synthetic --num-samples 100
```

GeoAI 弱标签样本，适合把真实影像跑通训练流程：

```bash
python scripts/generate_sample.py \
  --mode weak-label \
  --image-a data/raw/time_a/xxx_A.tif \
  --image-b data/raw/time_b/xxx_B_aligned.tif
```

真实矢量标签样本，适合正式监督训练：

```bash
python scripts/generate_sample.py \
  --mode vector-label \
  --image-a data/raw/time_a/xxx_A.tif \
  --image-b data/raw/time_b/xxx_B_aligned.tif \
  --vector-label labels/change.geojson
```

## 5. 训练自定义模型

```bash
python scripts/train.py --model siamese_unet --epochs 30 --batch-size 4
```

如果提示样本不足，请先确认：

```text
data/samples/time_a/*.tif
data/samples/time_b/*.tif
data/samples/labels/*.tif
```

三个目录数量必须一致，且至少 2 对样本。

## 6. 自训练模型推理

```bash
python scripts/predict.py \
  --engine cdd \
  --image-a data/raw/time_a/xxx_A.tif \
  --image-b data/raw/time_b/xxx_B_aligned.tif \
  --model data/models/best_model.pth \
  --model-name siamese_unet \
  --visualize
```

## 7. 启动 Web 界面

```bash
python start.py
```

或分别启动：

```bash
python -m uvicorn backend.main:app --reload --port 8000
cd frontend
npm run dev
```

访问 http://localhost:5173。

## 8. 运行测试

```bash
pytest tests/ -v
cd frontend
npm run build
```
