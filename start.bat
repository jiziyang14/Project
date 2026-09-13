@echo off
setlocal
rem ============================================================
rem  Quick-start script: launch / stop the AI learning companion local service.
rem  Usage:
rem    start.bat                 -> start on default port 8000
rem    start.bat 8001            -> start on port 8001
rem    start.bat stop            -> stop service on default port 8000 and release it
rem    start.bat stop 8001       -> stop service on port 8001 and release it
rem  Port priority for start: argument > APP_PORT > PORT > 8000
rem ============================================================

set "ARG1=%~1"
set "ARG2=%~2"

if /i "%ARG1%"=="stop" goto stop

rem ---- resolve the port: argument > interactive prompt > default 8000 ----
set "PORT=%ARG1%"
if not "%PORT%"=="" goto port_resolved
set /p "PORT=Enter port (default 8000, press Enter to accept): "
if "%PORT%"=="" set "PORT=8000"
:port_resolved

rem ---- locate a Python that has fastapi installed ----
rem NOTE: ALT must be set OUTSIDE the parenthesized blocks, otherwise %ALT%
rem is expanded (empty) at parse time and the probe is silently skipped.
set "ALT=C:\Users\young\AppData\Local\Programs\Python\Python312\python.exe"
set "PY="
rem 1) target the known-good Python 3.12 directly via the launcher
py -3.12 -c "import fastapi" >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
    rem 2) plain python on PATH (only useful if it is a real install)
    python -c "import fastapi" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    rem 3) fall back to the known-good interpreter by absolute path
    if exist "%ALT%" (
        "%ALT%" -c "import fastapi" >nul 2>&1 && set "PY=\"%ALT%\""
    )
)
if not defined PY (
    echo.
    echo [Error] Could not find a Python with fastapi installed.
    echo [Error] Tried: py -3.12, python, and %ALT%
    echo [Error] Install dependencies in your Python first:  pip install -r requirements.txt
    pause
    endlocal
    exit /b 1
)

echo [Start] Using Python: %PY%
echo [Start] Launching AI Learning Companion on port %PORT% ...
echo [Start] Starting up... I will beep when it is ready.
powershell -noprofile -command "[console]::Beep(660,120)"
echo.

rem ---- if the port is already occupied by a stale/old service, release it first ----
rem (so you never get "port in use" that you cannot clear; we start fresh with the latest code)
set "BUSY="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /c:":%PORT% " ^| findstr /c:"LISTENING"') do (
    echo [Start] Port %PORT% is occupied by PID %%a - releasing it automatically ...
    taskkill /f /pid %%a >nul 2>&1
    set "BUSY=1"
)
if defined BUSY (
    echo [Start] Released. Waiting for the port to free up ...
    ping -n 3 127.0.0.1 >nul
)
echo.

start "" /b %PY% -m backend.app

rem ---- wait until the service responds ----
for /l %%i in (1,1,40) do (
    ping -n 2 127.0.0.1 >nul
    powershell -noprofile -command "try{$r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%PORT%/' -TimeoutSec 1; if($r.StatusCode -eq 200){exit 0}}catch{exit 1}"
    if not errorlevel 1 goto ready
)
echo.
echo [Error] Service did not become ready in time. Check the output above.
pause
exit /b 1

:ready
rem Distinct ascending 3-note chime so you know startup is complete
powershell -noprofile -command "[console]::Beep(880,150);Start-Sleep -m 90;[console]::Beep(1174,150);Start-Sleep -m 90;[console]::Beep(1568,450)"
echo.
echo    ############################################################
echo    #   READY!!!  All services are up.                          #
echo    #   Open  http://127.0.0.1:%PORT%  in your browser.                 #
echo    ############################################################
echo.
echo [Start] Service is UP. Press Ctrl+C to stop.
echo.

:waitloop
ping -n 3 127.0.0.1 >nul
powershell -noprofile -command "try{$r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%PORT%/' -TimeoutSec 1; exit 0}catch{exit 1}"
if errorlevel 1 (
    echo.
    echo [Stop] Service has stopped.
    pause
    exit /b 0
)
goto waitloop

:stop
set "PORT=%ARG2%"
if "%PORT%"=="" set "PORT=8000"
echo [Stop] Releasing port %PORT% ...
set "FOUND="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    echo [Stop] Stopping PID %%a on port %PORT% ...
    taskkill /f /pid %%a >nul 2>&1 && echo [Stop] Released PID %%a
    set "FOUND=1"
)
if not defined FOUND echo [Stop] No service is listening on port %PORT%.
echo [Stop] Done.
endlocal
exit /b