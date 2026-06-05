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
$DefaultOpenCvBin = Join-Path $env:LOCALAPPDATA "rtk\opencv-4.13.0\opencv\build\x64\vc16\bin"
if (Test-Path $DefaultOpenCvBin) {
    $env:PATH = "$DefaultOpenCvBin;$env:PATH"
}
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $PythonCommand) {
    $PythonUserScripts = python -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $PythonUserScripts -ne "" -and
        (Test-Path $PythonUserScripts)) {
        $env:PATH = "$PythonUserScripts;$env:PATH"
    }
}

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
