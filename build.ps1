param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "build"

$OpenCVDir = $env:OpenCV_DIR
$DefaultOpenCvDir = Join-Path $env:LOCALAPPDATA "rtk\opencv-4.13.0\opencv\build"
if (($null -eq $OpenCVDir -or $OpenCVDir -eq "") -and
    (Test-Path $DefaultOpenCvDir)) {
    $OpenCVDir = $DefaultOpenCvDir
}

$CvlibPlatform = "linux"
$CvlibLibraryName = "libcvlib_core.a"
if ($env:OS -eq "Windows_NT") {
    $CvlibPlatform = "msvc"
    $CvlibLibraryName = "cvlib_core.lib"
} elseif ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::OSX)) {
    $CvlibPlatform = "macos"
}

$ConfigLower = $Config.ToLowerInvariant()
$CvlibLibrary = Join-Path $ScriptDir `
    "thirdparty\cvlib\lib\$CvlibPlatform\$ConfigLower\$CvlibLibraryName"
if (!(Test-Path $CvlibLibrary)) {
    & (Join-Path $ScriptDir "bundle_cvlib.ps1") `
        -Config $Config `
        -Platform $CvlibPlatform
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$ConfigureArgs = @(
    "-S", $ScriptDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DMVO_CVLIB_PLATFORM=$CvlibPlatform"
)

if ($OpenCVDir -ne "") {
    $ConfigureArgs += "-DOpenCV_DIR=$OpenCVDir"
}
if ($env:MVO_RERUN_SDK_URL -ne "") {
    $ConfigureArgs += "-DMVO_RERUN_SDK_URL=$env:MVO_RERUN_SDK_URL"
}

& cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
