@echo off
setlocal

REM Sentinel-2 正式处理启动脚本（Windows / Conda python12）
REM 在资源管理器双击本文件，或在终端执行：resNet\sentinel2\run_sentinel2.bat

set "CONDA_ROOT=D:\Program\Conda\Miniconda3"
set "CONDA_ENV=D:\Program\Conda\envs\python12"
set "WORK_DIR=%~dp0..\.."
set "INPUT_SAFE=%WORK_DIR%\resNet\sentinel2\00_extracted_safe"
set "OUTPUT_DIR=%WORK_DIR%\resNet\sentinel2\out"
set "CHECKPOINT=%WORK_DIR%\models\Classification\checkpoints\best_model.pth"

REM GPU 可用时改成 cuda；CPU 环境请保持 cpu。
set "DEVICE=cpu"
REM GPU 可将批大小尝试改为 16、32 或更大；CPU 建议保持 4。
set "BATCH_SIZE=4"

if not exist "%INPUT_SAFE%" (
    echo ERROR: Sentinel-2 SAFE directory was not found:
    echo %INPUT_SAFE%
    pause
    exit /b 1
)

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV%"
if errorlevel 1 (
    echo ERROR: Cannot activate conda environment: %CONDA_ENV%
    pause
    exit /b 1
)

set "MODEL_ARG="
if exist "%CHECKPOINT%" set "MODEL_ARG=--checkpoint "%CHECKPOINT%""

echo.
echo Starting Sentinel-2 pipeline in python12 environment...
echo Input : %INPUT_SAFE%
echo Output: %OUTPUT_DIR%
echo Device: %DEVICE%
echo.

python "%WORK_DIR%\resNet\sentinel2\sentinel2_pipeline.py" ^
  --input-safe "%INPUT_SAFE%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --bbox 97.38 36.48 103.77 39.73 ^
  --tile-size 192 ^
  --batch-size %BATCH_SIZE% ^
  --device %DEVICE% ^
  %MODEL_ARG%

if errorlevel 1 (
    echo.
    echo Pipeline stopped with an error. See the messages above.
) else (
    echo.
    echo Pipeline completed. Results are in: %OUTPUT_DIR%
)
pause
