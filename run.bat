@echo off
setlocal

:: A script to set up the Python environment and run the LLM Proxy on Windows.

:: --- Configuration ---
set VENV_DIR=.venv
set HOST=0.0.0.0
set PORT=8001

:: --- Check if Python is installed ---
echo Checking for Python installation...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ERROR: 'python' command not found.
    echo Please install Python 3.9+ and ensure it's added to your system's PATH.
    pause
    exit /b 1
)
echo Python found.

:: --- Set up Virtual Environment ---
if not exist "%VENV_DIR%\" (
    echo Virtual environment not found. Creating one at '%VENV_DIR%'...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Existing virtual environment found.
)

:: --- Activate Virtual Environment ---
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: --- Install Dependencies ---
echo.
echo --- Installing Dependencies ---

echo Installing packages from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements from requirements.txt.
    pause
    exit /b 1
)

echo Installing GiNZA model 'ja_ginza'...
pip install ja_ginza
if %errorlevel% neq 0 (
    echo ERROR: Failed to install ja_ginza model.
    pause
    exit /b 1
)

echo Running Argos Translate setup...
python setup_argos.py
if %errorlevel% neq 0 (
    echo ERROR: Argos Translate setup script failed.
    pause
    exit /b 1
)

echo All Python dependencies installed successfully.

:: --- Check for JMdict (Critical Prerequisite) ---
echo.
echo --- Checking for JMdict file ---
if not exist "jmdict.json" (
    echo.
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo !! CRITICAL ERROR: 'jmdict.json' not found.                !!
    echo !! This file is required and must be generated manually once.  !!
    echo !!                                                           !!
    echo !! To fix this:                                              !!
    echo !! 1. Download 'JMdict_e.gz' from the JMdict Project Page.   !!
    echo !! 2. Unzip it to get the 'JMdict_e' file in this directory. !!
    echo !! 3. Run this command: python parse_jmdict.py             !!
    echo !! 4. Re-run this setup script.                              !!
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pause
    exit /b 1
)

echo Found 'jmdict.json'. Prerequisites are met.


:: --- Launch the Application ---
echo.
echo --- Setup Complete ---
echo Launching the LLM Linguistic Enhancement Proxy...
echo Access the dashboard at: http://%HOST%:%PORT%
echo Press CTRL+C to stop the server.

uvicorn run:app --host %HOST% --port %PORT%

endlocal
