param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",

    [switch]$InstallOpenCV,

    [switch]$InstallRerun,

    [switch]$InstallAll,

    [switch]$NoInstall,

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "build"
$OpenCVVersion = if ([string]::IsNullOrWhiteSpace($env:MVO_OPENCV_VERSION)) {
    "4.13.0"
} else {
    $env:MVO_OPENCV_VERSION
}
$RerunVersion = if ([string]::IsNullOrWhiteSpace($env:MVO_RERUN_VERSION)) {
    "0.33.0"
} else {
    $env:MVO_RERUN_VERSION
}
$DependencyRoot = if (![string]::IsNullOrWhiteSpace($env:MVO_DEPS_ROOT)) {
    $env:MVO_DEPS_ROOT
} elseif (![string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    Join-Path $env:LOCALAPPDATA "MVO"
} elseif (![string]::IsNullOrWhiteSpace($env:XDG_CACHE_HOME)) {
    Join-Path $env:XDG_CACHE_HOME "mvo"
} elseif (![string]::IsNullOrWhiteSpace($env:HOME)) {
    Join-Path (Join-Path $env:HOME ".cache") "mvo"
} else {
    Join-Path $ScriptDir ".deps"
}

$InstallOptionSeen = $PSBoundParameters.ContainsKey("InstallOpenCV") -or
    $PSBoundParameters.ContainsKey("InstallRerun") -or
    $PSBoundParameters.ContainsKey("InstallAll") -or
    $PSBoundParameters.ContainsKey("NoInstall")

if (!$InstallOptionSeen -or $InstallAll) {
    $InstallOpenCV = $true
    $InstallRerun = $true
}
if ($NoInstall) {
    $InstallOpenCV = $false
    $InstallRerun = $false
}

$OpenCVDir = $env:OpenCV_DIR
$DefaultOpenCvRoot = Join-Path $DependencyRoot "opencv-$OpenCVVersion"
$DefaultOpenCvDir = Join-Path $DefaultOpenCvRoot "opencv\build"
if (($null -eq $OpenCVDir -or $OpenCVDir -eq "") -and
    (Test-Path (Join-Path $DefaultOpenCvDir "OpenCVConfig.cmake"))) {
    $OpenCVDir = $DefaultOpenCvDir
}

function Show-OpenCVPrerequisites {
    [Console]::Error.WriteLine(@"
OpenCV was not found.

Install OpenCV, then rerun .\build.ps1:
  Windows PowerShell: .\build.ps1 -InstallOpenCV
  Git Bash:           ./build.sh --install-opencv

If OpenCV is already installed in a custom location, set OpenCV_DIR to the
directory containing OpenCVConfig.cmake.

Set MVO_DEPS_ROOT to change the default per-user dependency install root.
"@)
}

function Show-RerunPrerequisites {
    [Console]::Error.WriteLine(@"
Rerun viewer was not found.

Install Rerun, then rerun .\build.ps1:
  Windows PowerShell: .\build.ps1 -InstallRerun
  Git Bash:           ./build.sh --install-rerun
  Python/pip:         py -m pip install --user rerun-sdk==$RerunVersion

The Rerun viewer version should match the C++ SDK version used by this project.
"@)
}

function Test-OpenCVAvailable {
    param([string]$CandidateDir)

    if ($CandidateDir -ne "" -and
        (Test-Path (Join-Path $CandidateDir "OpenCVConfig.cmake"))) {
        return $true
    }

    if (Get-Command pkg-config -ErrorAction SilentlyContinue) {
        & pkg-config --exists opencv4
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    return $false
}

function Test-RerunAvailable {
    return $null -ne (Get-Command rerun -ErrorAction SilentlyContinue)
}

function Test-CvlibAvailable {
    param(
        [string]$Platform,
        [string]$ConfigName,
        [string[]]$LibraryNames
    )

    $LibDir = Join-Path $ScriptDir "thirdparty\cvlib\lib\$Platform\$($ConfigName.ToLowerInvariant())"
    foreach ($LibraryName in $LibraryNames) {
        if (Test-Path (Join-Path $LibDir $LibraryName)) {
            return $true
        }
    }

    return $false
}

function Add-PathDir {
    param([string]$PathDir)

    if ($PathDir -ne "" -and
        (Test-Path $PathDir) -and
        (($env:PATH -split [IO.Path]::PathSeparator) -notcontains $PathDir)) {
        $env:PATH = "$PathDir$([IO.Path]::PathSeparator)$env:PATH"
    }
}

function Add-PythonUserScriptsToPath {
    foreach ($PythonCmd in @("py", "python", "python3")) {
        if (Get-Command $PythonCmd -ErrorAction SilentlyContinue) {
            $PreviousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $UserBase = & $PythonCmd -c "import site; print(site.USER_BASE)" 2>$null
                $PythonExitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $PreviousErrorActionPreference
            }
            if ($PythonExitCode -eq 0 -and $UserBase -ne "") {
                Add-PathDir (Join-Path $UserBase "Scripts")
                Add-PathDir (Join-Path $UserBase "bin")
            }
        }
    }

    if ($env:APPDATA -ne "") {
        Get-ChildItem -Directory -Path (Join-Path $env:APPDATA "Python") -Filter "Python*" -ErrorAction SilentlyContinue |
            ForEach-Object { Add-PathDir (Join-Path $_.FullName "Scripts") }
    }
}

function Install-OpenCVPackage {
    $InstallRoot = $DefaultOpenCvRoot
    $OpenCVUrl = if ([string]::IsNullOrWhiteSpace($env:MVO_OPENCV_WINDOWS_URL)) {
        "https://github.com/opencv/opencv/releases/download/$OpenCVVersion/opencv-$OpenCVVersion-windows.exe"
    } else {
        $env:MVO_OPENCV_WINDOWS_URL
    }
    $InstallerPath = Join-Path $env:TEMP "opencv-$OpenCVVersion-windows.exe"

    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
    Write-Host "Downloading OpenCV $OpenCVVersion..."
    Invoke-WebRequest -Uri $OpenCVUrl -OutFile $InstallerPath

    Write-Host "Extracting OpenCV to $InstallRoot..."
    $Process = Start-Process `
        -FilePath $InstallerPath `
        -ArgumentList @("-o$InstallRoot", "-y") `
        -Wait `
        -PassThru `
        -WindowStyle Hidden
    if ($Process.ExitCode -ne 0) {
        throw "OpenCV installer failed with exit code $($Process.ExitCode)."
    }

    if (!(Test-Path (Join-Path $DefaultOpenCvDir "OpenCVConfig.cmake"))) {
        throw "OpenCV was installed, but OpenCVConfig.cmake was not found under $DefaultOpenCvDir."
    }
}

function Install-RerunPackage {
    $PipArgs = @(
        "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--user",
        "rerun-sdk==$RerunVersion"
    )

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py @PipArgs
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python @PipArgs
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        & python3 @PipArgs
    } else {
        Show-RerunPrerequisites
        exit 1
    }

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Add-PythonUserScriptsToPath
}

Add-PythonUserScriptsToPath

if ($InstallOpenCV -and !(Test-OpenCVAvailable -CandidateDir $OpenCVDir)) {
    Install-OpenCVPackage
    $OpenCVDir = $DefaultOpenCvDir
    $env:OpenCV_DIR = $OpenCVDir
}

if ($InstallRerun -and !(Test-RerunAvailable)) {
    Install-RerunPackage
}

if ($SkipBuild) {
    exit 0
}

if (!(Test-OpenCVAvailable -CandidateDir $OpenCVDir)) {
    Show-OpenCVPrerequisites
    exit 1
}

if (!(Test-RerunAvailable)) {
    Show-RerunPrerequisites
    exit 1
}

$CvlibPlatform = "linux"
$CvlibLibraryNames = @("libcvlib_core.a", "libcvlib_core.so")
if ($env:OS -eq "Windows_NT") {
    $CvlibPlatform = "msvc"
    $CvlibLibraryNames = @("cvlib_core.lib")
} elseif ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::OSX)) {
    $CvlibPlatform = "macos"
    $CvlibLibraryNames = @("libcvlib_core.a", "libcvlib_core.dylib")
}

if (!(Test-CvlibAvailable `
        -Platform $CvlibPlatform `
        -ConfigName $Config `
        -LibraryNames $CvlibLibraryNames)) {
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
if (-not [string]::IsNullOrWhiteSpace($env:MVO_RERUN_SDK_URL)) {
    $ConfigureArgs += "-DMVO_RERUN_SDK_URL=$env:MVO_RERUN_SDK_URL"
}

& cmake --log-level=WARNING @ConfigureArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
