@echo off
REM ============================================================
REM TradingBot V3 - Market Data Prefetch Task
REM ============================================================
REM This script is designed to run via Windows Task Scheduler
REM
REM Recommended schedule:
REM   - Name: TradingBot_Prefetch_Weekly
REM   - Trigger: Weekly, Sunday at 23:00
REM   - Action: Start this batch file
REM
REM To create the scheduled task via command line:
REM   schtasks /create /tn "TradingBot_Prefetch_Weekly" /tr "C:\Users\tkado\Documents\Tradingbot_V3\scripts\prefetch_task.bat" /sc weekly /d SUN /st 23:00
REM
REM To run manually:
REM   scripts\prefetch_task.bat
REM ============================================================

REM Configuration
set PYTHON_PATH=C:\Python313\python.exe
set PROJECT_DIR=C:\Users\tkado\Documents\Tradingbot_V3
set LOG_DIR=%PROJECT_DIR%\data\logs

REM Create log directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Set date for log file
for /f "tokens=1-3 delims=/" %%a in ('echo %date%') do (
    set LOGDATE=%%c%%a%%b
)
for /f "tokens=1-2 delims=:" %%a in ('echo %time%') do (
    set LOGTIME=%%a%%b
)
set LOGFILE=%LOG_DIR%\prefetch_%LOGDATE%_%LOGTIME%.log

REM Log start
echo ============================================================ >> "%LOGFILE%"
echo TradingBot V3 - Market Data Prefetch >> "%LOGFILE%"
echo Started: %date% %time% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Run prefetch script (weekly data, all markets, no crypto)
echo Running prefetch... >> "%LOGFILE%"
"%PYTHON_PATH%" scripts\prefetch_market_data.py --weekly --clear-expired >> "%LOGFILE%" 2>&1

REM Check result
if %ERRORLEVEL% EQU 0 (
    echo. >> "%LOGFILE%"
    echo Prefetch completed successfully >> "%LOGFILE%"
) else (
    echo. >> "%LOGFILE%"
    echo ERROR: Prefetch failed with code %ERRORLEVEL% >> "%LOGFILE%"
)

REM Log end
echo. >> "%LOGFILE%"
echo Finished: %date% %time% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

exit /b %ERRORLEVEL%
