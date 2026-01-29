#!/bin/bash
# =============================================================================
# TradingBot V5 - Déploiement Raspberry Pi 4
# =============================================================================
#
# Ce script configure un Raspberry Pi 4 pour exécuter le bot 24/7.
#
# Prérequis:
#   - Raspberry Pi 4 (4GB RAM recommandé)
#   - Raspberry Pi OS (64-bit recommandé)
#   - Connexion internet stable
#
# Usage:
#   1. Copier le projet sur le Pi
#   2. chmod +x scripts/deploy_raspberry.sh
#   3. ./scripts/deploy_raspberry.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "  TradingBot V5 - Raspberry Pi Deployment"
echo "=============================================="
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier qu'on est sur un Pi
if [[ ! -f /proc/device-tree/model ]]; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
fi

# Variables
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
SERVICE_NAME="tradingbot"

echo "Project directory: $PROJECT_DIR"
echo ""

# =============================================================================
# 1. Mise à jour système
# =============================================================================
echo -e "${GREEN}[1/7] Updating system...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# =============================================================================
# 2. Installation des dépendances système
# =============================================================================
echo -e "${GREEN}[2/7] Installing system dependencies...${NC}"
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libatlas-base-dev \
    gfortran

# =============================================================================
# 3. Création de l'environnement virtuel
# =============================================================================
echo -e "${GREEN}[3/7] Creating Python virtual environment...${NC}"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Mettre à jour pip
pip install --upgrade pip wheel setuptools

# =============================================================================
# 4. Installation des dépendances Python
# =============================================================================
echo -e "${GREEN}[4/7] Installing Python dependencies...${NC}"
pip install -r "$PROJECT_DIR/requirements.txt"

# Dépendances spécifiques au Pi (versions optimisées ARM)
pip install numpy --upgrade
pip install pandas --upgrade

# =============================================================================
# 5. Configuration du fichier .env
# =============================================================================
echo -e "${GREEN}[5/7] Configuring environment...${NC}"
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    if [[ -f "$PROJECT_DIR/.env.example" ]]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        echo -e "${YELLOW}Created .env from .env.example - PLEASE EDIT WITH YOUR API KEYS${NC}"
    fi
fi

# =============================================================================
# 6. Installation de Tailscale (accès distant)
# =============================================================================
echo -e "${GREEN}[6/7] Installing Tailscale for remote access...${NC}"
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  IMPORTANT: Configure Tailscale now!${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "Run: sudo tailscale up"
    echo "Then login with your Tailscale account"
    echo ""
    echo "After login, your Pi will have a Tailscale IP (e.g., 100.x.x.x)"
    echo "You can access the dashboard from anywhere at:"
    echo "  http://100.x.x.x:8080"
    echo ""
fi

# =============================================================================
# 7. Création du service systemd
# =============================================================================
echo -e "${GREEN}[7/7] Creating systemd service...${NC}"

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=TradingBot V5 - Autonomous Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$VENV_DIR/bin/python webapp.py
Restart=always
RestartSec=10

# Logging
StandardOutput=append:$PROJECT_DIR/logs/tradingbot.log
StandardError=append:$PROJECT_DIR/logs/tradingbot_error.log

# Limits
MemoryLimit=1G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

# Créer le dossier de logs
mkdir -p "$PROJECT_DIR/logs"

# Recharger systemd
sudo systemctl daemon-reload

echo ""
echo "=============================================="
echo -e "${GREEN}  Deployment Complete!${NC}"
echo "=============================================="
echo ""
echo "Commands disponibles:"
echo ""
echo "  Démarrer le bot:"
echo "    sudo systemctl start $SERVICE_NAME"
echo ""
echo "  Arrêter le bot:"
echo "    sudo systemctl stop $SERVICE_NAME"
echo ""
echo "  Voir les logs:"
echo "    tail -f $PROJECT_DIR/logs/tradingbot.log"
echo "    journalctl -u $SERVICE_NAME -f"
echo ""
echo "  Activer au démarrage:"
echo "    sudo systemctl enable $SERVICE_NAME"
echo ""
echo "  Status:"
echo "    sudo systemctl status $SERVICE_NAME"
echo ""
echo "=============================================="
echo "  Accès distant avec Tailscale"
echo "=============================================="
echo ""
if command -v tailscale &> /dev/null; then
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "Non configuré")
    echo "  Tailscale IP: $TAILSCALE_IP"
    if [[ "$TAILSCALE_IP" != "Non configuré" ]]; then
        echo "  Dashboard URL: http://$TAILSCALE_IP:8080"
    fi
else
    echo "  Tailscale non installé"
fi
echo ""
echo "  Depuis votre PC/Mac/téléphone:"
echo "  1. Installez Tailscale: https://tailscale.com/download"
echo "  2. Connectez-vous au même compte"
echo "  3. Accédez à http://<tailscale-ip>:8080"
echo ""
echo "=============================================="
