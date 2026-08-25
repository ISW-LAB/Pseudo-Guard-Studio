@echo off
REM ==================================================================================
REM  PG-Label - install the training pack (PyTorch + Ultralytics).
REM
REM  The app labels, auto-labels and exports without this. You only need it for the
REM  "Train on few-label" and "Run cycle" buttons. It creates ONE folder
REM  (%LOCALAPPDATA%\PG-Label\gpu-env) and registers it with the app.
REM
REM  Download size: ~2.5 GB with CUDA, ~250 MB with --cuda cpu.
REM  Requires a normal Python 3.9-3.12 (64-bit) on this PC:
REM       winget install -e --id Python.Python.3.11
REM
REM  Options (passed straight through):
REM       install_training_pack.bat --cuda cpu                 no NVIDIA GPU
REM       install_training_pack.bat --python "C:\Python311\python.exe"
REM       install_training_pack.bat --upgrade                  refresh an existing pack
REM ==================================================================================
setlocal
cd /d "%~dp0"

if exist "%~dp0PG-Label.exe" (
  set "APP=%~dp0PG-Label.exe"
) else if exist "%~dp0dist\PG-Label\PG-Label.exe" (
  set "APP=%~dp0dist\PG-Label\PG-Label.exe"
) else (
  echo.
  echo   PG-Label.exe was not found next to this script.
  echo   Copy this file into the PG-Label install folder, or use the Start-menu
  echo   entry "Install training pack" instead.
  echo.
  pause
  exit /b 1
)

"%APP%" --install-gpu-pack %*
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" (
  echo   Done. Restart PG-Label - the Train button is now enabled.
) else (
  echo   The training pack was not installed ^(exit code %RC%^) - see the messages above.
)
pause
exit /b %RC%
