param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $repoRoot "requirements-runtime-windows.txt"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "Creating project environment at $venvPath"
    & python -m venv $venvPath --system-site-packages
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python was not created: $pythonPath"
}

if (-not $SkipInstall) {
    Write-Host "Installing pinned runtime dependencies"
    & $pythonPath -m pip install -r $requirementsPath
}

Write-Host "Running environment smoke check"
& $pythonPath (Join-Path $repoRoot "scripts\environment_doctor.py") `
    --package torch `
    --package torchaudio `
    --package funasr `
    --package modelscope `
    --package wespeaker `
    --package onnxruntime-gpu `
    --artifact (Join-Path $repoRoot "datasetA.zip")

Write-Host "Runtime ready. Activate with: . .\scripts\activate_runtime.ps1"
