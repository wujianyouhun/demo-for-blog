@echo off
setlocal

REM data.tif 的 ResNet 单图测试；结果保存到 resNet\data_resnet_result.json。
set "CONDA_ROOT=D:\Program\Conda\Miniconda3"
set "CONDA_ENV=D:\Program\Conda\envs\python12"
set "WORK_DIR=%~dp0.."
set "DATA_TIF=%WORK_DIR%\resNet\data.tif"
set "RESULT_JSON=%WORK_DIR%\resNet\data_resnet_result.json"
set "CHECKPOINT=%WORK_DIR%\models\Classification\checkpoints\best_model.pth"

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV%"
if errorlevel 1 exit /b 1

set "MODEL_ARG="
if exist "%CHECKPOINT%" set "MODEL_ARG=--checkpoint "%CHECKPOINT%""

REM 默认假设第 1/2/3/4 波段为 蓝/绿/红/近红外，因此用 3/2/1 作为 RGB。
python "%WORK_DIR%\resNet\resnet_inference_example.py" --geotiff "%DATA_TIF%" --bands 3 2 1 --scale 0.0001 --device cpu --output "%RESULT_JSON%" %MODEL_ARG%

pause
