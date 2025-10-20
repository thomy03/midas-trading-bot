@echo off
REM Windows batch script to run the market screener

echo ========================================
echo Market Screener
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo ========================================
echo Choose an option:
echo ========================================
echo 1. Run screening once
echo 2. Start scheduler (auto mode)
echo 3. Test notifications
echo 4. Screen a specific symbol
echo 5. View recent alerts
echo 6. Run tests
echo 7. Launch Dashboard (Web UI)
echo ========================================
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo Running screening...
    python main.py run
) else if "%choice%"=="2" (
    echo Starting scheduler...
    echo Press Ctrl+C to stop
    python main.py schedule
) else if "%choice%"=="3" (
    echo Testing notifications...
    python main.py test
) else if "%choice%"=="4" (
    set /p symbol="Enter stock symbol (e.g., AAPL): "
    echo Screening %symbol%...
    python main.py screen --symbol %symbol%
) else if "%choice%"=="5" (
    set /p days="Number of days to look back (default 7): "
    if "%days%"=="" set days=7
    python main.py alerts --days %days%
) else if "%choice%"=="6" (
    echo Running tests...
    python tests\test_basic.py
) else if "%choice%"=="7" (
    echo Starting Dashboard...
    echo Opening browser at http://localhost:8501
    streamlit run dashboard.py
) else (
    echo Invalid choice!
)

echo.
pause
