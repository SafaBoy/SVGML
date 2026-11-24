param([switch]$Yes)

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (-not $Yes) {
  $reply = Read-Host "Stop SVGML containers and remove local build artifacts? [y/N]"
  if ($reply -notmatch '^(y|Y|yes|YES)$') {
    Write-Host "Aborted. Nothing was removed."
    exit 0
  }
}

Write-Host "Stopping SVGML project containers..."
if (Get-Command docker -ErrorAction SilentlyContinue) {
  & docker compose -f docker-compose.yml -f docker-compose.gpu.yml down --remove-orphans | Out-Host
} else {
  Write-Host "Docker is unavailable; skipping container cleanup."
}

Write-Host "Removing local build and runtime artifacts..."
$paths = @(
  '.run',
  '*.log',
  '*.pid',
  'frontend/dist',
  'frontend/node_modules',
  'frontend/tsconfig.tsbuildinfo',
  'backend-rust/target',
  'backend-go/backend-go',
  'bin',
  'dist',
  'ml-worker/.venv',
  'data/gallery.json'
)
foreach ($path in $paths) {
  Remove-Item -Path (Join-Path $RootDir $path) -Recurse -Force -ErrorAction SilentlyContinue
}

Get-ChildItem -Path $RootDir -Directory -Recurse -Force -Filter '__pycache__' -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch '\\\.git\\' } |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $RootDir -File -Recurse -Force -Include '*.pyc', '*.pyo', '*.pyd' -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch '\\\.git\\' } |
  Remove-Item -Force -ErrorAction SilentlyContinue

$embeddedDirs = @(
  'backend-go/embedded/frontend',
  'backend-go/embedded/ml-worker',
  'backend-go/embedded/bin'
)
foreach ($dir in $embeddedDirs) {
  $fullDir = Join-Path $RootDir $dir
  if (Test-Path $fullDir) {
    Get-ChildItem -Path $fullDir -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -ne '.gitkeep' -and $_.Name -ne '.keep' } |
      Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  }
}

Write-Host "Desk clean. Project containers stopped and local artifacts removed."
Write-Host "Model caches, local env files, volumes and unrelated Docker data were preserved."
