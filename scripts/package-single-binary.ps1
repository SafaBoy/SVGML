$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DistDir = Join-Path $RootDir "dist"
$EmbedDir = Join-Path $RootDir "backend-go\embedded"

Write-Host "========================================================"
Write-Host "  SVGML Studio — Windows & Linux Single-Binary Packager"
Write-Host "========================================================"

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
if (Test-Path $EmbedDir) { Remove-Item -Recurse -Force $EmbedDir }
New-Item -ItemType Directory -Force -Path (Join-Path $EmbedDir "frontend") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $EmbedDir "ml-worker") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $EmbedDir "bin") | Out-Null

# 1. Build React Frontend
Write-Host "`n[1/4] Building React Studio Frontend..."
Push-Location (Join-Path $RootDir "frontend")
if (-not (Test-Path "node_modules")) { npm install }
npm run build
Pop-Location
Copy-Item -Recurse -Force (Join-Path $RootDir "frontend\dist\*") (Join-Path $EmbedDir "frontend\")

# 2. Stage Python ML Worker
Write-Host "`n[2/4] Staging Python PyTorch/CUDA ML Worker..."
Copy-Item -Force (Join-Path $RootDir "ml-worker\app.py") (Join-Path $EmbedDir "ml-worker\app.py")
Copy-Item -Force (Join-Path $RootDir "ml-worker\requirements.txt") (Join-Path $EmbedDir "ml-worker\requirements.txt")

# 3. Build Rust vtracer Engine for Windows
Write-Host "`n[3/4] Building Rust SVG & vtracer Engine for Windows..."
Push-Location (Join-Path $RootDir "backend-rust")
cargo build --release
Copy-Item -Force "target\release\svgml-rust-engine.exe" (Join-Path $EmbedDir "bin\svgml-rust-engine-windows-amd64.exe")
Pop-Location

# 4. Build Windows & Linux Single Binaries
Write-Host "`n[4/4] Compiling Single Binaries..."
Push-Location (Join-Path $RootDir "backend-go")
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"
go build -trimpath -ldflags="-s -w" -o (Join-Path $DistDir "svgml-studio-windows-amd64.exe") .

$env:GOOS = "linux"
$env:GOARCH = "amd64"
go build -trimpath -ldflags="-s -w" -o (Join-Path $DistDir "svgml-studio-linux-amd64") .
Pop-Location

Write-Host "`n✅ Created single binaries in $DistDir:"
Get-ChildItem $DistDir
