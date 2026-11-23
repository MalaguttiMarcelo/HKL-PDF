<#
Build the HKL-PDF Windows program and its installer.

    powershell -ExecutionPolicy Bypass -File installer\build.ps1

Steps
  1. Build environment on a LOCAL disk (default %LOCALAPPDATA%\HKL-PDF-build),
     with the pinned packages of installer\requirements-build.txt.
  2. GPU manifest: the exact CuPy / CUDA wheels (with checksums) that the
     installed program downloads when the user asks for GPU support.
  3. PyInstaller folder build (HKL-PDF.spec).
  4. Remove CuPy and any CUDA library from the program folder: GPU support
     is downloaded on request, never shipped.
  5. Smoke test: the built program compiles its kernels (--warmup).
     With -TestGpu: also download the GPU support and run --gpu-check.
  6. Inno Setup: installer-output\HKL-PDF-Setup-<version>.exe

Options
  -BuildRoot <dir>   where the environment and build output go (local disk!)
  -Recreate          delete and recreate the build environment
  -NoInstaller       stop after the smoke test
  -TestGpu           test the downloadable GPU support with the built program
#>
param(
    [string]$BuildRoot = (Join-Path $env:LOCALAPPDATA "HKL-PDF-build"),
    [switch]$Recreate,
    [switch]$NoInstaller,
    [switch]$TestGpu
)

# "Continue": warnings that pip / PyInstaller print on stderr must not stop
# the script (Windows PowerShell turns them into errors); every native step
# is checked with $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $BuildRoot "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Dist = Join-Path $BuildRoot "dist"
$App = Join-Path $Dist "HKL-PDF"
$Exe = Join-Path $App "HKL-PDF.exe"
$Log = Join-Path $env:LOCALAPPDATA "HKL-PDF\hkl-pdf.log"

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }

# Run the built program (no console: its messages go to the log file).
function Invoke-Built($arguments) {
    $before = if (Test-Path $Log) { (Get-Item $Log).Length } else { 0 }
    $p = Start-Process -FilePath $Exe -ArgumentList $arguments -Wait -PassThru
    $text = if (Test-Path $Log) { [System.IO.File]::ReadAllText($Log) } else { "" }
    $new = if ($text.Length -gt $before) { $text.Substring([int]$before) } else { "" }
    Write-Host $new
    return @{ Code = $p.ExitCode; Text = $new }
}

# Version from pdf_fitting/__init__.py (single source)
$Version = (Select-String -Path (Join-Path $Repo "pdf_fitting\__init__.py") -Pattern '^__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
Write-Host "HKL-PDF $Version  ·  build folder $BuildRoot"

# 1. Build environment -------------------------------------------------------
Step "1/6 Build environment"
if ($Recreate -and (Test-Path $Venv)) { Remove-Item -Recurse -Force $Venv }
if (-not (Test-Path $Py)) {
    $base = (& py -3.14 -c "import sys; print(sys.executable)").Trim()
    New-Item -ItemType Directory -Force $BuildRoot | Out-Null
    & $base -m venv $Venv
    if ($LASTEXITCODE) { throw "Could not create the virtual environment." }
}
& $Py -m pip install --upgrade pip --quiet
& $Py -m pip install -r (Join-Path $PSScriptRoot "requirements-build.txt") --quiet
if ($LASTEXITCODE) { throw "pip install of the build requirements failed." }
# HKL-PDF itself is not installed: PyInstaller takes it from the repository.

# 2. GPU manifest -------------------------------------------------------------
Step "2/6 GPU download manifest"
& $Py (Join-Path $PSScriptRoot "make_gpu_manifest.py")
if ($LASTEXITCODE) { throw "Could not write pdf_fitting\gpu_packages.json (internet needed)." }

# 3. PyInstaller ---------------------------------------------------------------
Step "3/6 PyInstaller (folder build)"
if (Test-Path $Dist) { Remove-Item -Recurse -Force $Dist }
& $Py -m PyInstaller (Join-Path $Repo "HKL-PDF.spec") --noconfirm --clean `
    --workpath (Join-Path $BuildRoot "work") --distpath $Dist
if ($LASTEXITCODE) { throw "PyInstaller failed." }

# 4. No GPU libraries in the program -----------------------------------------
Step "4/6 Remove CuPy / CUDA from the program folder"
$Internal = Join-Path $App "_internal"
$gpuItems = Get-ChildItem $Internal | Where-Object {
    $_.Name -match '^(cupy|cupyx|cupy_backends|cuda|nvidia|fastrlock)$' -or
    $_.Name -match '^(cupy_cuda|fastrlock|cuda_pathfinder|cuda_toolkit|nvidia_).*\.dist-info$' -or
    $_.Name -match '^(cudart|nvrtc|cublas|cusparse|cusolver|cufft|curand|nvjitlink|nvJitLink).*\.dll$'
}
foreach ($item in $gpuItems) { Remove-Item -Recurse -Force $item.FullName }
$files = Get-ChildItem $App -Recurse -File
$mainMB = [math]::Round((($files | Measure-Object Length -Sum).Sum) / 1MB)
Write-Host "Removed $($gpuItems.Count) GPU items. Program folder: $mainMB MB in $($files.Count) files."

# 5. Smoke tests --------------------------------------------------------------
Step "5/6 Smoke test (--warmup)"
$r = Invoke-Built "--warmup"
if ($r.Code -ne 0 -or $r.Text -notmatch "\[WARMUP\] done") { throw "The built program failed the warm-up test (see $Log)." }
if ($r.Text -match "failed") { throw "A warm-up example failed (see $Log)." }

if ($TestGpu) {
    Step "5b Downloadable GPU support"
    & $Py -c "import sys; sys.path.insert(0, r'$Repo'); from pdf_fitting import gpu_install as g; g.installed() or g.install(lambda t,d,n: None); print('GPU support ready in', g.gpu_dir())"
    if ($LASTEXITCODE) { throw "Downloading the GPU support failed." }
    $r = Invoke-Built "--gpu-check"
    if ($r.Code -ne 0) { throw "The built program could not use the downloaded GPU support (see $Log)." }
}

if ($NoInstaller) { Write-Host "Program built in $App"; exit 0 }

# 6. Installer ----------------------------------------------------------------
Step "6/6 Inno Setup"
$iscc = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found. Install it from https://jrsoftware.org/isinfo.php" }
$gpuMB = [math]::Round(((Get-Content (Join-Path $Repo "pdf_fitting\gpu_packages.json") -Raw | ConvertFrom-Json).wheels | Measure-Object size -Sum).Sum / 1MB)
& $iscc "/DMyAppVersion=$Version" "/DSourceDir=$App" "/DGpuDownloadMB=$gpuMB" (Join-Path $PSScriptRoot "HKL-PDF.iss")
if ($LASTEXITCODE) { throw "Inno Setup failed." }
$setup = Join-Path $Repo "installer-output\HKL-PDF-Setup-$Version.exe"
Write-Host ""
Write-Host ("Installer: {0}  ({1:N0} MB)" -f $setup, ((Get-Item $setup).Length / 1MB)) -ForegroundColor Green
