#!/bin/bash
# Deployment script for Midas BULL Optimization
# Usage: ./deploy.sh [ssh-key-path]

VPS_HOST="root@46.225.58.233"
VPS_PATH="/root/tradingbot-github"
SSH_KEY="${1:-/home/node/.ssh/id_ed25519}"

echo "=== Midas BULL Optimization Deployment ==="
echo "VPS: $VPS_HOST"
echo "Path: $VPS_PATH"
echo "SSH Key: $SSH_KEY"
echo ""

# Check SSH connection
echo "Testing SSH connection..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$VPS_HOST" "echo 'Connected'" 2>/dev/null; then
    echo "ERROR: SSH connection failed!"
    echo ""
    echo "Please ensure the SSH key is authorized on the VPS:"
    echo "  1. Copy public key to VPS authorized_keys"
    echo "  2. Or use: ssh-copy-id -i $SSH_KEY $VPS_HOST"
    exit 1
fi

echo "SSH connection OK"
echo ""

# Create directories
echo "Creating directories..."
ssh -i "$SSH_KEY" "$VPS_HOST" "mkdir -p $VPS_PATH/src/brokers $VPS_PATH/config"

# Copy broker files
echo "Copying broker module..."
scp -i "$SSH_KEY" src/brokers/__init__.py "$VPS_HOST:$VPS_PATH/src/brokers/"
scp -i "$SSH_KEY" src/brokers/ib_broker.py "$VPS_HOST:$VPS_PATH/src/brokers/"
scp -i "$SSH_KEY" src/brokers/paper_trader.py "$VPS_HOST:$VPS_PATH/src/brokers/"

# Copy scoring files
echo "Copying scoring module..."
scp -i "$SSH_KEY" src/scoring/bull_optimizer.py "$VPS_HOST:$VPS_PATH/src/scoring/"
scp -i "$SSH_KEY" src/scoring/adaptive_scorer_patch.py "$VPS_HOST:$VPS_PATH/src/scoring/"

# Copy agent files
echo "Copying agent module..."
scp -i "$SSH_KEY" src/agents/reasoning_engine_patch.py "$VPS_HOST:$VPS_PATH/src/agents/"

# Copy config
echo "Copying configuration..."
scp -i "$SSH_KEY" config/pillar_weights.json "$VPS_HOST:$VPS_PATH/config/"

# Copy documentation
echo "Copying documentation..."
scp -i "$SSH_KEY" SCORING_SYSTEM_UPDATE.md "$VPS_HOST:$VPS_PATH/"
scp -i "$SSH_KEY" README.md "$VPS_HOST:$VPS_PATH/BULL_OPTIMIZATION_README.md"

# Install dependencies
echo "Installing ib_insync..."
ssh -i "$SSH_KEY" "$VPS_HOST" "pip install ib_insync --quiet"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. SSH to VPS: ssh -i $SSH_KEY $VPS_HOST"
echo "2. Navigate to project: cd $VPS_PATH"
echo "3. Run backtest: python -m src.backtest.runner --start 2020-01-01 --end 2025-01-01"
echo "4. Review SCORING_SYSTEM_UPDATE.md for integration details"
