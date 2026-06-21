# 快速上手

## 环境安装

```bash
# Conda（推荐）
conda env create -f environment.yml
conda activate change-detection

# 或 pip
pip install -r requirements.txt
```

## 快速体验流程

### 1. 生成模拟数据（无需网络）

```bash
python scripts/generate_sample.py
```

这会在 `data/samples/` 下生成 100 对模拟双时相影像和变化标签，可直接用于训练。

### 2. 训练模型

```bash
python scripts/train.py --model siamese_unet --epochs 30 --batch-size 8
```

### 3. 下载真实数据

```bash
python scripts/download_data.py --region beijing --date-a 2022-06-01 --date-b 2023-06-01
```

### 4. 执行变化检测

```bash
python scripts/predict.py \
    --image-a data/raw/time_a/xxx.tif \
    --image-b data/raw/time_b/xxx_aligned.tif \
    --model data/models/best_model.pth \
    --visualize
```

### 5. 启动 Web 界面

```bash
# 后端
cd backend && uvicorn main:app --reload

# 前端
cd frontend && npm install && npm run dev
```

访问 http://localhost:5173

## 运行测试

```bash
# 全部测试
pytest tests/ -v

# 仅模型测试
pytest tests/test_models.py -v

# 仅 API 测试
pytest tests/test_api.py -v
```
