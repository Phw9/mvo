param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [Nullable[int]]$MaxFrames = $null,
    [switch]$NoBa,
    [string]$InputConfig = "",
    [string]$InputType = "",
    [string]$InputPath = "",
    [string]$ParameterDir = "",
    [string]$Calib = "",
    [string]$RerunSave = "",
    [switch]$RerunSpawn,
    [switch]$NoRerun,
    [switch]$DebugGeometry
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$OpenCVVersion = if ([string]::IsNullOrWhiteSpace($env:MVO_OPENCV_VERSION)) {
    "4.13.0"
} else {
    $env:MVO_OPENCV_VERSION
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

$DefaultOpenCvBin = Join-Path $env:LOCALAPPDATA "rtk\opencv-$OpenCVVersion\opencv\build\x64\vc16\bin"
Add-PathDir $DefaultOpenCvBin
Add-PythonUserScriptsToPath

$Exe = Join-Path $ScriptDir "build\$Config\mvo_cvlib.exe"
if (!(Test-Path $Exe)) {
    $Exe = Join-Path $ScriptDir "build\mvo_cvlib.exe"
}
if (!(Test-Path $Exe)) {
    throw "mvo_cvlib executable not found. Run .\build.ps1 first."
}

$RunArgs = @("--no-gui")
if ($null -ne $MaxFrames) {
    $RunArgs += @("--max-frames", "$MaxFrames")
}
if ($NoBa) {
    $RunArgs += "--no-ba"
}
if ($InputConfig -ne "") {
    $RunArgs += @("--input-config", $InputConfig)
}
if ($InputType -ne "") {
    $RunArgs += @("--input-type", $InputType)
}
if ($InputPath -ne "") {
    $RunArgs += @("--input-path", $InputPath)
}
if ($ParameterDir -ne "") {
    $RunArgs += @("--parameter-dir", $ParameterDir)
}
if ($Calib -ne "") {
    $RunArgs += @("--calib", $Calib)
}
if ($DebugGeometry) {
    $RunArgs += "--debug-geometry"
}
if ($NoRerun) {
    $RunArgs += "--no-rerun"
} elseif ($RerunSave -ne "") {
    $RunArgs += @("--rerun-save", $RerunSave)
} else {
    $RunArgs += "--rerun-spawn"
}

& $Exe @RunArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
