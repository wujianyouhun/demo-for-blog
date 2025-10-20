@echo off
chcp 65001 >nul
echo RTSP流媒体转换器启动脚本
echo =========================

echo 检查FFmpeg是否安装...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到FFmpeg，请先安装FFmpeg并添加到PATH环境变量中
    echo 下载地址: https://ffmpeg.org/download.html
    pause
    exit /b 1
)

echo FFmpeg已安装，继续启动...
echo.

echo 启动RTSP到Web流媒体转换器...
python rtsp_to_web.py

pause