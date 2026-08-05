param(
    [switch]$DryRun,
    [switch]$MockObstacle,
    [int]$Cam = -1,
    [int]$TrafficCam = -1,
    [string]$SerialPort = ""
)

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$main = Join-Path $PSScriptRoot "integrated_traffic_obstacle.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python venv not found: $python"
}

$runArgs = @($main)
if ($DryRun) {
    $runArgs += "--dry-run"
}
if ($MockObstacle) {
    $runArgs += "--mock-obstacle"
}
if ($Cam -ge 0) {
    $runArgs += @("--cam", $Cam)
}
if ($TrafficCam -ge 0) {
    $runArgs += @("--traffic-cam", $TrafficCam)
}
if ($SerialPort) {
    $runArgs += @("--serial-port", $SerialPort)
}

& $python @runArgs
exit $LASTEXITCODE
