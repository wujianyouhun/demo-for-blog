param(
    [ValidateSet('audit', 'train', 'evaluate', 'infer')]
    [string]$Action = 'audit',
    [string]$Python = 'python',
    [string]$Device = 'auto',
    [int]$Epochs = 60,
    [string]$Image = '',
    [string]$Experiment = 'landcover_6class_optimized'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$datasetPath = Join-Path $PSScriptRoot 'data/landcover_6class'
$originalCheckpoint = Join-Path $projectRoot 'models/deeplabv3/landcover_6class_best.pth'
if ($Experiment -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Experiment 仅支持字母、数字、下划线与短横线。' }
$experimentPath = Join-Path $projectRoot "models/deeplabv3/$Experiment"
$reportPath = Join-Path $PSScriptRoot "out/optimization/$Experiment"
$optimizedCheckpoint = Join-Path $experimentPath 'best.pth'

switch ($Action) {
    'audit' {
        & $Python (Join-Path $PSScriptRoot 'audit_classification.py') --dataset-dir $datasetPath --checkpoint $originalCheckpoint --output (Join-Path $reportPath 'audit.json')
    }
    'train' {
        & $Python (Join-Path $PSScriptRoot 'train_optimized.py') --dataset-dir $datasetPath --output-dir $experimentPath --init-checkpoint $originalCheckpoint --epochs $Epochs --device $Device
    }
    'evaluate' {
        & $Python (Join-Path $PSScriptRoot 'evaluate_checkpoint.py') --dataset-dir $datasetPath --checkpoint $optimizedCheckpoint --output (Join-Path $reportPath 'evaluation.json') --device $Device
    }
    'infer' {
        if (-not $Image) { throw 'infer 需要 -Image 指定影像路径。' }
        & $Python (Join-Path $PSScriptRoot 'infer_optimized.py') --image $Image --checkpoint $optimizedCheckpoint --output (Join-Path $reportPath 'classification.tif') --confidence (Join-Path $reportPath 'max_probability.tif') --device $Device
    }
}
if ($LASTEXITCODE -ne 0) { throw "执行失败，退出代码：$LASTEXITCODE" }
