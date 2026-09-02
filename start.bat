@echo off
echo ================================================
echo   Aeterna Avatar Backend — Starting...
echo ================================================
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

:: Start server
echo.
echo Starting Aeterna Avatar API on http://localhost:8000
echo.
python server.py
