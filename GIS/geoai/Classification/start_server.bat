@echo off
echo ============================================
echo   GeoAI 遥感图像分类系统 - 启动服务
echo ============================================
echo.
call conda activate geoai
if errorlevel 1 (
    echo [ERROR] 激活 conda 环境 geoai 失败
    pause
    exit /b 1
)
echo [OK] conda 环境 geoai 已激活
echo.
echo [INFO] 启动 FastAPI 服务 (http://localhost:8000)
echo [INFO] 按 Ctrl+C 停止服务
echo.
cd /d %~dp0
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
