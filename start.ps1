# Quick start script: launch / stop the AI learning companion local service.
# Usage:
#   .\start.ps1                        -> start on default port 8000
#   .\start.ps1 -Port 8001             -> start on port 8001
#   .\start.ps1 -Stop                  -> stop service on default port 8000 and release it
#   .\start.ps1 -Stop -Port 8001       -> stop service on port 8001 and release it
# Port priority for start: -Port > APP_PORT > PORT > 8000

param([int]$Port, [switch]$Stop)

# 双击 / 右键运行友好：若尚未以“绕过执行策略”的方式运行，则自动重新拉起一次，
# 避免受执行策略限制无法启动。（用环境变量防止无限循环；端口 / 停止参数会透传。）
if (-not $env:AI_COPILOT_STARTED) {
    $env:AI_COPILOT_STARTED = "1"
    if ($PSCommandPath) {
        $invoke = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
        if (-not $Port -or $Port -eq 0) { $Port = 8000 }
        $invoke += "-Port", "$Port"
        if ($Stop) { $invoke += "-Stop" }
        & powershell.exe @invoke
        exit $LASTEXITCODE
    }
}

# 端口：-Port 参数 > 交互输入（双击运行时请在此输入）> 默认 8000
if (-not $PSBoundParameters.ContainsKey('Port')) {
    $input = Read-Host "Enter port (default 8000, press Enter to accept)"
    if ($input -and $input -match '^\d+$') { $Port = [int]$input } else { $Port = 8000 }
} elseif (-not $Port) {
    $Port = 8000
}

# Find the absolute path of a Python that has fastapi (avoids the Windows Store stub).
# NOTE: must NOT use `& $array` for a command array (it is treated as one command name);
# resolve each candidate to its real python.exe path via sys.executable.
function Get-PyCommand {
    $candidates = @(
        @('py', '-3.12'),
        @('python'),
        @('C:\Users\young\AppData\Local\Programs\Python\Python312\python.exe')
    )
    foreach ($cmd in $candidates) {
        $exe = $cmd[0]
        $rest = @()
        if ($cmd.Count -gt 1) { $rest = $cmd[1..($cmd.Count - 1)] }
        $found = (Get-Command $exe -ErrorAction SilentlyContinue) -ne $null -or (Test-Path $exe)
        if (-not $found) { continue }
        try {
            $sysExe = (& $exe @rest -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
            if ($LASTEXITCODE -eq 0 -and $sysExe -and (Test-Path $sysExe)) {
                & $sysExe -c "import fastapi" 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) { return $sysExe.Trim() }
            }
        } catch { }
    }
    return $null
}

function Stop-PortService {
    param([int]$PortToRelease)
    $conns = Get-NetTCPConnection -LocalPort $PortToRelease -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "[Stop] No service is listening on port $PortToRelease." -ForegroundColor Yellow
        return
    }
    $procIds = $conns.OwningProcess | Sort-Object -Unique
    $killed = $false
    foreach ($procId in $procIds) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[Stop] Stopping PID $procId ($($proc.ProcessName)) on port $PortToRelease ..." -ForegroundColor Cyan
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            $killed = $true
        }
    }
    if ($killed) {
        Write-Host "[Stop] Port $PortToRelease released." -ForegroundColor Green
    }
}

if ($Stop) {
    Stop-PortService $Port
    exit 0
}

$pyCmd = Get-PyCommand
if (-not $pyCmd) {
    Write-Host ""
    Write-Host "[Error] Could not find a Python with fastapi installed." -ForegroundColor Red
    Write-Host "[Error] Install dependencies first:  pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

$env:APP_PORT = "$Port"

Write-Host "[Start] Launching AI Learning Companion on port $Port ..." -ForegroundColor Cyan
Write-Host "[Start] Starting up... I will beep when it is ready." -ForegroundColor Cyan
[console]::Beep(660, 120)
Write-Host ""

# 若端口被旧/残留服务占用，自动释放，确保以最新代码启动（根治“端口被占用又无法解除”）
$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conns) {
    $ids = $conns.OwningProcess | Sort-Object -Unique
    foreach ($id in $ids) {
        $p = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($p) {
            Write-Host "[Start] Port $Port is occupied by PID $id - releasing it automatically ..." -ForegroundColor Yellow
            Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 600
}

# 后台启动服务，前台轮询端口就绪（输出重定向到临时日志，确保可靠启动）
# Get-PyCommand returns a single absolute python.exe path (a string).
$exe = $pyCmd
$argsList = @('-m', 'backend.app')

$outLog = Join-Path $env:TEMP "ai-copilot-server.log"
$errLog = Join-Path $env:TEMP "ai-copilot-server.err.log"
$proc = Start-Process -FilePath $exe -ArgumentList $argsList -WorkingDirectory (Get-Location).Path -PassThru -NoNewWindow -RedirectStandardOutput $outLog -RedirectStandardError $errLog

$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 1
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}

if ($ready) {
    # 上升三连音，提示所有服务已就绪
    [console]::Beep(880, 150)
    Start-Sleep -Milliseconds 90
    [console]::Beep(1174, 150)
    Start-Sleep -Milliseconds 90
    [console]::Beep(1568, 450)
    Write-Host ""
    Write-Host "############################################################" -ForegroundColor Green
    Write-Host "#   READY!!!  All services are up.                          #" -ForegroundColor Green
    Write-Host "#   Open  http://127.0.0.1:$Port  in your browser.                 #" -ForegroundColor Green
    Write-Host "############################################################" -ForegroundColor Green
    Write-Host ""
    Write-Host "[Start] Service is UP. Press Ctrl+C to stop." -ForegroundColor Green
    Write-Host ""
    Wait-Process -Id $proc.Id
} else {
    Write-Host "[Error] Service did not become ready in time. Check the logs above." -ForegroundColor Red
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    exit 1
}