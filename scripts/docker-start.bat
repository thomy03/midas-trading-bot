@echo off
REM TradingBot V3 - Docker Start Script (Windows)
REM
REM Usage:
REM   docker-start.bat              - Start all services
REM   docker-start.bat dashboard    - Start dashboard only
REM   docker-start.bat api          - Start API only
REM   docker-start.bat build        - Build images only

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

echo ================================
echo TradingBot V3 - Docker Manager
echo ================================

REM Check if .env exists
if not exist ".env" (
    echo Warning: .env file not found
    echo Creating from .env.example...
    copy .env.example .env
    echo Please edit .env with your configuration
)

REM Check Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running
    exit /b 1
)

REM Parse command
if "%1"=="" goto all
if "%1"=="build" goto build
if "%1"=="dashboard" goto dashboard
if "%1"=="api" goto api
if "%1"=="scheduler" goto scheduler
if "%1"=="all" goto all
if "%1"=="stop" goto stop
if "%1"=="logs" goto logs
if "%1"=="status" goto status
goto usage

:build
echo Building Docker images...
docker-compose build
echo Build complete!
goto end

:dashboard
echo Starting Dashboard service...
docker-compose up -d dashboard
echo Dashboard started at http://localhost:8501
goto end

:api
echo Starting API service...
docker-compose up -d api
echo API started at http://localhost:8000
echo API docs: http://localhost:8000/docs
goto end

:scheduler
echo Starting Scheduler service...
docker-compose up -d scheduler
echo Scheduler started
goto end

:all
echo Starting all services...
docker-compose up -d
echo.
echo All services started!
echo.
echo Services:
echo   - Dashboard: http://localhost:8501
echo   - API:       http://localhost:8000
echo   - API Docs:  http://localhost:8000/docs
echo.
echo View logs: docker-compose logs -f
goto end

:stop
echo Stopping all services...
docker-compose down
echo All services stopped
goto end

:logs
docker-compose logs -f %2
goto end

:status
docker-compose ps
goto end

:usage
echo Usage: %0 {build^|dashboard^|api^|scheduler^|all^|stop^|logs^|status}
echo.
echo Commands:
echo   build      - Build Docker images
echo   dashboard  - Start dashboard only
echo   api        - Start API only
echo   scheduler  - Start scheduler only
echo   all        - Start all services (default)
echo   stop       - Stop all services
echo   logs       - View logs (optionally specify service)
echo   status     - Show service status
exit /b 1

:end
endlocal
