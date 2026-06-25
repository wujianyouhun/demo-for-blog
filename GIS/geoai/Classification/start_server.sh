#!/bin/bash
echo "============================================"
echo "  GeoAI 遥感图像分类系统 - 启动服务"
echo "============================================"
source activate geoai || conda activate geoai
cd "$(dirname "$0")"
echo "[INFO] 启动 FastAPI 服务 (http://localhost:8000)"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
