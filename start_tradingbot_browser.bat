@echo off
title TradingBot V4 - Agent Autonome
color 0A

echo ========================================
echo    TradingBot V4 - Demarrage + Browser
echo ========================================
echo.

REM Aller dans le repertoire du projet
cd /d "%~dp0"

REM Tuer les processus existants sur le port 8080
echo [INFO] Verification du port 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080" ^| findstr "LISTENING"') do (
    echo [INFO] Fermeture du processus %%a sur le port 8080...
    taskkill /F /PID %%a >nul 2>&1
)

REM Verifier .env
if not exist ".env" (
    echo [WARN] Fichier .env manquant - creez-le a partir de .env.example
)

echo [INFO] Demarrage de la webapp...
echo [INFO] Le navigateur s'ouvrira dans 3 secondes...
echo.

REM Ouvrir le navigateur apres un delai (en arriere-plan)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8080"

REM Lancer la webapp (bloquant)
python webapp.py

pause
