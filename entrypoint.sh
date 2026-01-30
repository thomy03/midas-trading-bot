#!/bin/bash
# TradingBot Entrypoint - Lance le dashboard ET le bot live

set -e

echo "================================================"
echo "   TradingBot V5 - Unified Startup"
echo "================================================"
echo "Mode: ${TRADING_MODE:-paper}"
echo ""

# Créer les dossiers de logs
mkdir -p /app/logs

# Fonction pour lancer le bot live
start_live_bot() {
    echo "[ENTRYPOINT] Starting live trading bot..."
    cd /app
    if [ "$TRADING_MODE" = "live" ]; then
        python run_agent.py --mode live > /app/logs/live.log 2>&1 &
    else
        python run_agent.py --mode live --paper > /app/logs/live.log 2>&1 &
    fi
    BOT_PID=$!
    echo "[ENTRYPOINT] Live bot started (PID: $BOT_PID)"
}

# Fonction pour arrêter le bot
stop_live_bot() {
    echo "[ENTRYPOINT] Stopping live bot..."
    pkill -f "run_agent.py" 2>/dev/null || true
}

# Gérer les signaux pour arrêt propre
trap 'stop_live_bot; exit 0' SIGTERM SIGINT

# Si AUTO_START_BOT=true, lancer le bot immédiatement
if [ "${AUTO_START_BOT:-false}" = "true" ]; then
    start_live_bot
fi

# Lancer le dashboard NiceGUI (foreground)
echo "[ENTRYPOINT] Starting dashboard..."
exec python webapp.py
