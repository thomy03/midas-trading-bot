@echo off
REM TradingBot V4 - Agent Launcher
REM Usage: Start-Agent.bat [mode] [options]
REM
REM Modes:
REM   test      - Test avec mock IBKR (defaut)
REM   discovery - Scan social + decouverte
REM   analysis  - Analyse LLM
REM   trading   - Screening + execution
REM   full      - Cycle quotidien complet
REM
REM Options:
REM   --mock     - Utiliser mock IBKR
REM   --docker   - Lancer via Docker
REM   --stop     - Arreter l'agent

setlocal EnableDelayedExpansion

REM Aller au repertoire du projet
cd /d "%~dp0.."

REM Mode par defaut
set MODE=test
set ARGS=

REM Parser les arguments
:parse_args
if "%~1"=="" goto run
if /i "%~1"=="test" set MODE=test
if /i "%~1"=="discovery" set MODE=discovery
if /i "%~1"=="analysis" set MODE=analysis
if /i "%~1"=="trading" set MODE=trading
if /i "%~1"=="full" set MODE=full
if /i "%~1"=="--mock" set ARGS=!ARGS! --mock
if /i "%~1"=="--docker" goto docker_mode
if /i "%~1"=="--stop" goto stop_agent
shift
goto parse_args

:run
echo.
echo ============================================================
echo                 TradingBot V4 - Agent
echo ============================================================
echo Mode: %MODE%
echo.

REM Activer le venv si disponible
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Lancer l'agent
python run_agent.py --mode %MODE% %ARGS%
goto end

:docker_mode
echo.
echo ============================================================
echo           TradingBot V4 - Agent (Docker)
echo ============================================================
echo Mode: %MODE%
echo.

set AGENT_MODE=%MODE%
docker-compose up agent
goto end

:stop_agent
echo.
echo Arret de l'agent...

REM Arreter les processus Python
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *run_agent*" 2>nul

REM Arreter le container Docker
docker stop tradingbot-agent 2>nul

echo Agent arrete.
goto end

:end
endlocal
