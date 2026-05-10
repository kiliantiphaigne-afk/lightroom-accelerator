@echo off
echo ============================================
echo   Lightroom Accelerator — Build .exe
echo ============================================
echo.

pip install pyinstaller >NUL 2>&1

echo Construction du .exe (cela peut prendre 1-2 minutes)...

pyinstaller ^
    --onefile ^
    --windowed ^
    --name "LightroomAccelerator" ^
    --add-data "core;core" ^
    main.py

if errorlevel 1 (
    echo.
    echo [ERREUR] Build echoue.
    pause
    exit /b 1
)

echo.
echo [OK] Le .exe est dans le dossier : dist\LightroomAccelerator.exe
echo Vous pouvez le copier n'importe ou — il est autonome.
echo.
pause
