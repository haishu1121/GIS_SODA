<#
Package source code plus the active OODA train/validation JSONL for a Linux
server. Raw GIS, canonical data, test data, checkpoints, local credentials,
and virtual environments are intentionally excluded.
#>
[CmdletBinding()]
param(
  [string]$OutputDirectory = "dist",
  [string]$BundleName = "gis-concept-lora-server"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = Join-Path $ProjectRoot $OutputDirectory
$StageRoot = Join-Path $OutputRoot $BundleName
$ArchivePath = Join-Path $OutputRoot "$BundleName.zip"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if (Test-Path -LiteralPath $StageRoot) { Remove-Item -LiteralPath $StageRoot -Recurse -Force }
if (Test-Path -LiteralPath $ArchivePath) { Remove-Item -LiteralPath $ArchivePath -Force }
New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null

$IncludePaths = @(
  "src",
  "scripts",
  "configs",
  "tests",
  "README.md",
  "CHANGELOG.md",
  "pyproject.toml",
  ".gitignore",
  "data/sft/gis-concept-v1/CURRENT_DATASET.md",
  "data/sft/gis-concept-v1/llm_augmented/train/anonymous_ooda_en.jsonl",
  "data/sft/gis-concept-v1/llm_augmented/validation/anonymous_ooda_en.jsonl",
  "data/sft/gis-concept-v1/llm_augmented/export_manifest.json"
)
foreach ($RelativePath in $IncludePaths) {
  $SourcePath = Join-Path $ProjectRoot $RelativePath
  if (-not (Test-Path -LiteralPath $SourcePath)) { throw "Required bundle input is missing: $RelativePath" }
  $DestinationPath = Join-Path $StageRoot $RelativePath
  if (Test-Path -LiteralPath $SourcePath -PathType Container) {
    Get-ChildItem -LiteralPath $SourcePath -Recurse -File | Where-Object {
      $_.FullName -notlike "*__pycache__*" -and
      $_.Extension -ne ".pyc" -and
      $_.FullName -notlike "*egg-info*"
    } | ForEach-Object {
      $ChildRelativePath = $_.FullName.Substring($SourcePath.Length).TrimStart([char[]]@('\', '/'))
      $ChildDestination = Join-Path $DestinationPath $ChildRelativePath
      New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ChildDestination) | Out-Null
      Copy-Item -LiteralPath $_.FullName -Destination $ChildDestination -Force
    }
  } else {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DestinationPath) | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force
  }
}

Compress-Archive -Path (Join-Path $StageRoot "*") -DestinationPath $ArchivePath -CompressionLevel Optimal
Remove-Item -LiteralPath $StageRoot -Recurse -Force
Write-Host "Created server bundle: $ArchivePath"
