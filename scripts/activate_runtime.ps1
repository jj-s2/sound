$repoRoot = Split-Path -Parent $PSScriptRoot
$activatePath = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path -LiteralPath $activatePath)) {
    throw "Project environment is missing. Run .\scripts\setup_runtime.ps1 first."
}
. $activatePath
