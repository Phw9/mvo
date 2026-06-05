param(
    [string]$Config = "Release",
    [string]$CvlibSourceDir = "",
    [string]$Generator = "",
    [string]$Platform = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($CvlibSourceDir -eq "") {
    $WorkspaceDir = Split-Path -Parent $ScriptDir
    $CvlibSourceDir = Join-Path $WorkspaceDir "cvlib\cpp"
}

if (!(Test-Path (Join-Path $CvlibSourceDir "CMakeLists.txt"))) {
    Write-Error "cvlib source not found: $CvlibSourceDir"
    exit 1
}

$BuildDir = Join-Path $ScriptDir "build\cvlib_package"
$ConfigureArgs = @(
    "-S", $CvlibSourceDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCVLIB_BUILD_TESTS=OFF",
    "-DCVLIB_BUILD_PYTHON=OFF"
)
if ($Generator -ne "") {
    $ConfigureArgs = @("-G", $Generator) + $ConfigureArgs
}

& cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$BundleRoot = Join-Path $ScriptDir "thirdparty\cvlib"
$IncludeDest = Join-Path $BundleRoot "include"
if (Test-Path $IncludeDest) {
    Remove-Item -LiteralPath $IncludeDest -Recurse -Force
}
New-Item -ItemType Directory -Force $IncludeDest | Out-Null
$CvlibIncludeSource = Join-Path $CvlibSourceDir "include\cvlib"
if (!(Test-Path $CvlibIncludeSource)) {
    $CvlibIncludeSource = Join-Path $CvlibSourceDir "include"
}
Copy-Item -Path (Join-Path $CvlibIncludeSource "*") `
    -Destination $IncludeDest -Recurse -Force

$LibraryNames = @(
    "cvlib_core.lib",
    "libcvlib_core.a",
    "libcvlib_core.so",
    "libcvlib_core.dylib"
)
$Library = Get-ChildItem -Recurse -File $BuildDir |
    Where-Object { $LibraryNames -contains $_.Name } |
    Sort-Object FullName |
    Select-Object -First 1
if ($null -eq $Library) {
    Write-Error "cvlib library was not produced under $BuildDir"
    exit 1
}

if ($Platform -eq "") {
    $Platform = "linux"
    if ($env:OS -eq "Windows_NT") {
        $Platform = "msvc"
    } elseif ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
            [System.Runtime.InteropServices.OSPlatform]::OSX)) {
        $Platform = "macos"
    }
}

$ConfigLower = $Config.ToLowerInvariant()
$LibDest = Join-Path $BundleRoot "lib\$Platform\$ConfigLower"
New-Item -ItemType Directory -Force $LibDest | Out-Null
Copy-Item -LiteralPath $Library.FullName -Destination $LibDest -Force

Write-Host "cvlib bundled:"
Write-Host "  headers: $IncludeDest"
Write-Host "  library: $(Join-Path $LibDest $Library.Name)"
