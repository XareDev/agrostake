@echo off
REM Build script for AgroStake GUI Application (Windows)
REM Usage: build.bat

echo.
echo ========================================
echo   AgroStake GUI Builder
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/4] Checking dependencies...
pip list | findstr PyQt6 >nul 2>&1
if errorlevel 1 (
    echo [*] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [OK] All dependencies are installed
)

echo.
echo [2/4] Checking PyInstaller...
pip list | findstr pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [*] Installing PyInstaller...
    pip install pyinstaller
) else (
    echo [OK] PyInstaller is installed
)

echo.
echo [3/4] Building executable...
echo.
pyinstaller agrostake_gui.spec

if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo [4/4] Build complete!
echo.
echo Output: dist\AgroStake\AgroStake.exe
echo.
pause
