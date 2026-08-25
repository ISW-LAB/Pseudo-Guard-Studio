@echo off
REM ==================================================================================
REM  PG-Label - build PG-Label.exe on Windows.  Double-click me.
REM
REM  Needs: Python 3.9+ (64-bit) from python.org or the Microsoft Store.
REM         Everything else (PyInstaller, Pillow) goes into a throwaway environment
REM         under packaging\build\ - your Python is left untouched.
REM
REM  Optional, for the setup.exe installer:
REM         winget install -e --id JRSoftware.InnoSetup
REM         then run:  build.bat --installer
REM
REM  All flags pass through to build.py (--installer, --no-training, --no-demo,
REM  --windowed, --clean).  See docs\WINDOWS.md.
REM ==================================================================================
setlocal
cd /d "%~dp0"

REM The py launcher is the reliable way to reach a real python.org install; plain
REM `python` on PATH may be the Store alias stub that opens the Store instead.
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")

%PY% --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo   Python was not found.  Install it, then run this again:
  echo.
  echo       winget install -e --id Python.Python.3.11
  echo.
  pause
  exit /b 1
)

%PY% "%~dp0build.py" %*
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
  echo   Build finished.  The app is in:  %~dp0dist\PG-Label\
) else (
  echo   Build FAILED ^(exit code %RC%^) - scroll up for the first error.
)
pause
exit /b %RC%
