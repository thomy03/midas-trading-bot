#!/bin/bash
# Midas V8 - VPS Deployment Script
# Run this ON the VPS (via SSH)
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/thomy03/midas-trading-bot/main/deploy.sh | bash
#   OR: copy this file to VPS and run: bash deploy.sh

set -e

REPO_URL="https://github.com/thomy03/midas-trading-bot.git"
INSTALL_DIR="/opt/midas"
BRANCH="main"

echo "=========================================="
echo "  MIDAS V8 - VPS Deployment"
echo "=========================================="

# 1. System updates
echo "[1/6] Updating system..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl

# 2. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[2/6] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installed. Log out and back in if 'docker' fails without sudo."
else
    echo "[2/6] Docker already installed ($(docker --version))"
fi

# 3. Install Docker Compose plugin if not present
if ! docker compose version &> /dev/null 2>&1; then
    echo "[3/6] Installing Docker Compose plugin..."
    sudo apt-get install -y -qq docker-compose-plugin
else
    echo "[3/6] Docker Compose already installed"
fi

# 4. Clone or update repo
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[4/6] Updating existing repo..."
    cd "$INSTALL_DIR"
    git fetch origin
    git reset --hard "origin/$BRANCH"
else
    echo "[4/6] Cloning repo..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$USER:$USER" "$INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 5. Setup .env
if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo "[5/6] Creating .env from template..."
    cp "$INSTALL_DIR/.env.production" "$INSTALL_DIR/.env"
    echo ""
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║  EDIT .env WITH YOUR REAL API KEYS NOW   ║"
    echo "  ║  nano $INSTALL_DIR/.env                  ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo ""
    read -p "  Press Enter after editing .env (or Ctrl+C to abort)..."
else
    echo "[5/6] .env already exists (keeping current keys)"
fi

# Create data directories
mkdir -p "$INSTALL_DIR"/{data,logs,models}
mkdir -p "$INSTALL_DIR"/data/{cache,grok,tickers,signals,discovery,sectors,social}

# 6. Build and start
echo "[6/6] Building Docker image (this takes 3-5 min first time)..."
cd "$INSTALL_DIR"
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "=========================================="
echo "  Midas V8 deployed successfully!"
echo "=========================================="
echo ""
echo "  Status:      docker compose -f docker-compose.prod.yml ps"
echo "  Agent logs:  docker compose -f docker-compose.prod.yml logs -f agent"
echo "  API health:  curl http://localhost:8000/health"
echo "  Stop:        docker compose -f docker-compose.prod.yml down"
echo "  Restart:     docker compose -f docker-compose.prod.yml restart"
echo ""
