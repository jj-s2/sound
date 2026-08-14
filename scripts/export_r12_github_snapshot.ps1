[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceRoot).Path

if (Test-Path -LiteralPath $Destination) {
    $existing = Get-ChildItem -LiteralPath $Destination -Force
    if ($existing.Count -ne 0) {
        throw 'Destination must be empty.'
    }
} else {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}
$destinationPath = (Resolve-Path -LiteralPath $Destination).Path

$exact = @(
    '.gitignore',
    'README.md',
    'requirements-runtime-windows.txt',
    'configs/r12_paraformer_train.example.yaml',
    'docs/r12/r12-train-and-publish.md',
    'scripts/export_r12_github_snapshot.ps1',
    'tests/test_export_r12_github_snapshot.py',
    'tests/test_r12_publish_contract.py',
    'xh202615/data.py',
    'xh202615/r12_dataa_augmented_split.py',
    'xh202615/r12_dataa_augmentation.py'
)
$tracked = & git -C $source ls-files
if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed.' }

$selected = $tracked | Where-Object {
    $_ -in $exact -or
    $_ -like 'xh202615/r12_asr_*.py' -or
    $_ -like 'scripts/r12_asr_*.py' -or
    $_ -like 'tests/test_r12_asr_*.py'
}

foreach ($relative in $selected) {
    $from = Join-Path $source $relative
    $to = Join-Path $destinationPath $relative
    $parent = Split-Path -Parent $to
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $from -Destination $to -Force
}

[pscustomobject]@{
    source = $source
    destination = $destinationPath
    copied_files = @($selected).Count
    files = @($selected)
} | ConvertTo-Json -Depth 2
