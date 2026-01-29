#!/bin/bash
# =============================================================================
# TradingBot V5 - Raspberry Pi Control Script
# =============================================================================
#
# Script simplifié pour contrôler le bot sur Raspberry Pi
#
# Usage:
#   ./pi_control.sh start    - Démarrer le bot
#   ./pi_control.sh stop     - Arrêter le bot
#   ./pi_control.sh restart  - Redémarrer le bot
#   ./pi_control.sh status   - Voir le statut
#   ./pi_control.sh logs     - Voir les logs en direct
#   ./pi_control.sh url      - Afficher l'URL d'accès
#
# =============================================================================

SERVICE_NAME="tradingbot"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Couleurs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

case "$1" in
    start)
        echo -e "${GREEN}Starting TradingBot...${NC}"
        sudo systemctl start $SERVICE_NAME
        sleep 2
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;

    stop)
        echo -e "${RED}Stopping TradingBot...${NC}"
        sudo systemctl stop $SERVICE_NAME
        ;;

    restart)
        echo -e "${YELLOW}Restarting TradingBot...${NC}"
        sudo systemctl restart $SERVICE_NAME
        sleep 2
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;

    status)
        echo -e "${BLUE}TradingBot Status:${NC}"
        sudo systemctl status $SERVICE_NAME --no-pager
        echo ""
        echo -e "${BLUE}Memory Usage:${NC}"
        free -h
        echo ""
        echo -e "${BLUE}CPU Temperature:${NC}"
        vcgencmd measure_temp 2>/dev/null || echo "N/A"
        ;;

    logs)
        echo -e "${BLUE}TradingBot Logs (Ctrl+C to exit):${NC}"
        tail -f "$PROJECT_DIR/logs/tradingbot.log"
        ;;

    errors)
        echo -e "${RED}TradingBot Error Logs:${NC}"
        tail -50 "$PROJECT_DIR/logs/tradingbot_error.log"
        ;;

    url)
        echo ""
        echo "=============================================="
        echo -e "${GREEN}  TradingBot Access URLs${NC}"
        echo "=============================================="
        echo ""

        # IP locale
        LOCAL_IP=$(hostname -I | awk '{print $1}')
        echo -e "  ${BLUE}Local (same network):${NC}"
        echo "    http://$LOCAL_IP:8080"
        echo ""

        # IP Tailscale
        if command -v tailscale &> /dev/null; then
            TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
            if [[ -n "$TAILSCALE_IP" ]]; then
                echo -e "  ${GREEN}Remote (via Tailscale):${NC}"
                echo "    http://$TAILSCALE_IP:8080"
                echo ""
                echo -e "  ${YELLOW}Tailscale Status:${NC}"
                tailscale status
            else
                echo -e "  ${RED}Tailscale not connected${NC}"
                echo "    Run: sudo tailscale up"
            fi
        else
            echo -e "  ${RED}Tailscale not installed${NC}"
        fi
        echo ""
        ;;

    enable)
        echo -e "${GREEN}Enabling TradingBot at boot...${NC}"
        sudo systemctl enable $SERVICE_NAME
        echo "Done! TradingBot will start automatically on boot."
        ;;

    disable)
        echo -e "${YELLOW}Disabling TradingBot at boot...${NC}"
        sudo systemctl disable $SERVICE_NAME
        ;;

    update)
        echo -e "${BLUE}Updating TradingBot...${NC}"
        cd "$PROJECT_DIR"
        git pull
        source venv/bin/activate
        pip install -r requirements.txt
        echo -e "${GREEN}Update complete! Restart with: ./pi_control.sh restart${NC}"
        ;;

    tailscale)
        echo -e "${BLUE}Configuring Tailscale...${NC}"
        if ! command -v tailscale &> /dev/null; then
            curl -fsSL https://tailscale.com/install.sh | sh
        fi
        sudo tailscale up
        echo ""
        $0 url
        ;;

    *)
        echo "TradingBot V5 - Raspberry Pi Control"
        echo ""
        echo "Usage: $0 {command}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the bot"
        echo "  stop      - Stop the bot"
        echo "  restart   - Restart the bot"
        echo "  status    - Show bot status"
        echo "  logs      - View live logs"
        echo "  errors    - View error logs"
        echo "  url       - Show access URLs"
        echo "  enable    - Enable autostart on boot"
        echo "  disable   - Disable autostart"
        echo "  update    - Update from git"
        echo "  tailscale - Configure Tailscale VPN"
        echo ""
        exit 1
        ;;
esac
