@echo off
echo ============================================
echo   Lightroom Accelerator — Installation
echo ============================================
echo.

:: Verifier Python
python --version 2>NUL
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo Telecharge Python sur https://www.python.org/downloads/
    echo Coche bien "Add Python to PATH" a l'installation.
    pause
    exit /b 1
)

echo [1/2] Installation des dependances Python...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERREUR] Installation echouee. Verifiez votre connexion internet.
    pause
    exit /b 1
)

echo.
echo [2/2] Installation terminee !
echo.
echo Pour lancer l'application :
echo   python main.py
echo.
echo Pour creer un .exe (optionnel) :
echo   build_exe.bat
echo.
pause
