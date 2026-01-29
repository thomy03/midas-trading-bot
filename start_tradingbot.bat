@echo off
title TradingBot V4 - Agent Autonome
color 0A

echo ========================================
echo    TradingBot V4 - Demarrage
echo ========================================
echo.

REM Aller dans le repertoire du projet
cd /d "%~dp0"

REM Tuer les processus Python existants sur le port 8080
echo [INFO] Verification du port 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080" ^| findstr "LISTENING"') do (
    echo [INFO] Fermeture du processus %%a sur le port 8080...
    taskkill /F /PID %%a >nul 2>&1
)

REM Verifier que .env existe
if not exist ".env" (
    echo [WARN] Fichier .env manquant!
    echo        Copiez .env.example vers .env et configurez vos API keys.
    echo.
)

echo [INFO] Demarrage de la webapp sur http://localhost:8080
echo [INFO] Ctrl+C pour arreter
echo ========================================
echo.

REM Lancer la webapp
python webapp.py

REM Si erreur
if errorlevel 1 (
    echo.
    echo [ERREUR] La webapp s'est arretee avec une erreur.
    pause
)
