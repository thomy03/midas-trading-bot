#!/usr/bin/env bash
# Midas Trading Bot - SSL Certificate Renewal Script
# This script renews Let's Encrypt SSL certificates via the certbot container
# and reloads Nginx to pick up the new certificates.
#
# Usage:
#   ./nginx/ssl-renew.sh
#
# Crontab (run twice daily as recommended by Let's Encrypt):
#   0 2,14 * * * /path/to/Midas/nginx/ssl-renew.sh >> /var/log/midas-ssl-renew.log 2>&1

set -euo pipefail

# Resolve project root (directory containing docker-compose.prod.yml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.prod.yml"

LOG_PREFIX="[midas-ssl-renew]"

log() {
    echo "${LOG_PREFIX} $(date '+%Y-%m-%d %H:%M:%S') $*"
}

# Verify compose file exists
if [[ ! -f "${COMPOSE_FILE}" ]]; then
    log "ERROR: docker-compose.prod.yml not found at ${COMPOSE_FILE}"
    exit 1
fi

log "Starting SSL certificate renewal..."

# Run certbot renewal via the certbot container
docker compose -f "${COMPOSE_FILE}" run --rm certbot renew \
    --webroot \
    --webroot-path=/var/www/certbot \
    --quiet \
    --deploy-hook "echo 'Certificate renewed successfully'"

RENEW_EXIT=$?

if [[ ${RENEW_EXIT} -eq 0 ]]; then
    log "Certbot renewal check completed successfully."

    # Reload Nginx to pick up any renewed certificates (graceful, no downtime)
    log "Reloading Nginx configuration..."
    docker compose -f "${COMPOSE_FILE}" exec -T nginx nginx -s reload

    if [[ $? -eq 0 ]]; then
        log "Nginx reloaded successfully."
    else
        log "WARNING: Nginx reload failed. You may need to restart the nginx container manually."
    fi
else
    log "ERROR: Certbot renewal failed with exit code ${RENEW_EXIT}."
    exit 1
fi

log "SSL renewal process complete."
