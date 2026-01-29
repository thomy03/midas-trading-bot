#!/bin/bash
# TradingBot V3 - Docker Start Script
#
# Usage:
#   ./scripts/docker-start.sh              # Start all services
#   ./scripts/docker-start.sh dashboard    # Start dashboard only
#   ./scripts/docker-start.sh api          # Start API only
#   ./scripts/docker-start.sh build        # Build images only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}TradingBot V3 - Docker Manager${NC}"
echo -e "${GREEN}================================${NC}"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your configuration${NC}"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

# Parse command
case "${1:-all}" in
    build)
        echo "Building Docker images..."
        docker-compose build
        echo -e "${GREEN}Build complete!${NC}"
        ;;
    dashboard)
        echo "Starting Dashboard service..."
        docker-compose up -d dashboard
        echo -e "${GREEN}Dashboard started at http://localhost:8501${NC}"
        ;;
    api)
        echo "Starting API service..."
        docker-compose up -d api
        echo -e "${GREEN}API started at http://localhost:8000${NC}"
        echo -e "API docs: http://localhost:8000/docs"
        ;;
    scheduler)
        echo "Starting Scheduler service..."
        docker-compose up -d scheduler
        echo -e "${GREEN}Scheduler started${NC}"
        ;;
    all)
        echo "Starting all services..."
        docker-compose up -d
        echo ""
        echo -e "${GREEN}All services started!${NC}"
        echo ""
        echo "Services:"
        echo "  - Dashboard: http://localhost:8501"
        echo "  - API:       http://localhost:8000"
        echo "  - API Docs:  http://localhost:8000/docs"
        echo ""
        echo "View logs: docker-compose logs -f"
        ;;
    stop)
        echo "Stopping all services..."
        docker-compose down
        echo -e "${GREEN}All services stopped${NC}"
        ;;
    logs)
        docker-compose logs -f ${2:-}
        ;;
    status)
        docker-compose ps
        ;;
    *)
        echo "Usage: $0 {build|dashboard|api|scheduler|all|stop|logs|status}"
        echo ""
        echo "Commands:"
        echo "  build      - Build Docker images"
        echo "  dashboard  - Start dashboard only"
        echo "  api        - Start API only"
        echo "  scheduler  - Start scheduler only"
        echo "  all        - Start all services (default)"
        echo "  stop       - Stop all services"
        echo "  logs       - View logs (optionally specify service)"
        echo "  status     - Show service status"
        exit 1
        ;;
esac
