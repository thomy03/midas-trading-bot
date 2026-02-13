#!/usr/bin/env bash
# =============================================================================
# Midas Trading Bot - Backup Script
# =============================================================================
# Creates timestamped backups of data/, models/, Docker volumes, and config.
# Implements rolling retention to automatically purge old backups.
#
# Usage:
#   ./scripts/backup.sh
#   ./scripts/backup.sh --destination /mnt/external/backups --retention-days 60
#
# Exit codes:
#   0 - Success (all components backed up)
#   1 - Partial failure (some components failed, others succeeded)
#   2 - Critical failure (backup dir creation failed, or no components backed up)
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DESTINATION="/backups/midas"
RETENTION_DAYS=30
LOG_FILE="/var/log/midas-backup.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_NAME="midas_backup_${TIMESTAMP}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --destination)
            DESTINATION="$2"
            shift 2
            ;;
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--destination /path/to/backup] [--retention-days N]"
            echo ""
            echo "Options:"
            echo "  --destination DIR    Backup root directory (default: /backups/midas)"
            echo "  --retention-days N   Delete backups older than N days (default: 30)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    local message="$*"
    local entry="[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] ${message}"
    echo "${entry}"
    # Attempt to write to log file; if not writable, skip silently
    echo "${entry}" >> "${LOG_FILE}" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Ensure backup destination exists
# ---------------------------------------------------------------------------
BACKUP_DIR="${DESTINATION}/${BACKUP_NAME}"

if ! mkdir -p "${BACKUP_DIR}" 2>/dev/null; then
    log "CRITICAL" "Failed to create backup directory: ${BACKUP_DIR}"
    exit 2
fi

log "INFO" "=========================================="
log "INFO" "Midas Backup Started"
log "INFO" "=========================================="
log "INFO" "Project directory : ${PROJECT_DIR}"
log "INFO" "Backup destination: ${BACKUP_DIR}"
log "INFO" "Retention         : ${RETENTION_DAYS} days"

# Track success/failure counts
TOTAL=0
SUCCESS=0
FAILED=0

# ---------------------------------------------------------------------------
# 1. Backup data/ directory
# ---------------------------------------------------------------------------
TOTAL=$((TOTAL + 1))
if [[ -d "${PROJECT_DIR}/data" ]]; then
    log "INFO" "Backing up data/ directory..."
    if tar -czf "${BACKUP_DIR}/data.tar.gz" -C "${PROJECT_DIR}" data 2>/dev/null; then
        DATA_SIZE=$(du -sh "${BACKUP_DIR}/data.tar.gz" 2>/dev/null | cut -f1)
        log "INFO" "data/ backup complete (${DATA_SIZE})"
        SUCCESS=$((SUCCESS + 1))
    else
        log "ERROR" "Failed to back up data/ directory"
        FAILED=$((FAILED + 1))
    fi
else
    log "WARN" "data/ directory not found at ${PROJECT_DIR}/data -- skipping"
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# 2. Backup models/ directory
# ---------------------------------------------------------------------------
TOTAL=$((TOTAL + 1))
if [[ -d "${PROJECT_DIR}/models" ]]; then
    log "INFO" "Backing up models/ directory..."
    if tar -czf "${BACKUP_DIR}/models.tar.gz" -C "${PROJECT_DIR}" models 2>/dev/null; then
        MODELS_SIZE=$(du -sh "${BACKUP_DIR}/models.tar.gz" 2>/dev/null | cut -f1)
        log "INFO" "models/ backup complete (${MODELS_SIZE})"
        SUCCESS=$((SUCCESS + 1))
    else
        log "ERROR" "Failed to back up models/ directory"
        FAILED=$((FAILED + 1))
    fi
else
    log "WARN" "models/ directory not found at ${PROJECT_DIR}/models -- skipping"
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# 3. Backup Docker volumes (if docker is available)
# ---------------------------------------------------------------------------
TOTAL=$((TOTAL + 1))
if command -v docker &>/dev/null; then
    log "INFO" "Docker detected -- backing up volumes..."
    DOCKER_BACKUP_DIR="${BACKUP_DIR}/docker_volumes"
    mkdir -p "${DOCKER_BACKUP_DIR}"

    VOLUME_ERRORS=0
    for VOLUME_NAME in prometheus-data grafana-data; do
        # Compose prepends the project name; try both patterns
        # Check if the volume exists (with or without project prefix)
        RESOLVED_VOLUME=""
        for CANDIDATE in "${VOLUME_NAME}" "midas_${VOLUME_NAME}" "midas-${VOLUME_NAME}"; do
            if docker volume inspect "${CANDIDATE}" &>/dev/null; then
                RESOLVED_VOLUME="${CANDIDATE}"
                break
            fi
        done

        if [[ -n "${RESOLVED_VOLUME}" ]]; then
            log "INFO" "  Dumping volume: ${RESOLVED_VOLUME}"
            if docker run --rm \
                -v "${RESOLVED_VOLUME}:/volume:ro" \
                -v "${DOCKER_BACKUP_DIR}:/backup" \
                alpine \
                tar -czf "/backup/${VOLUME_NAME}.tar.gz" -C /volume . 2>/dev/null; then
                VOL_SIZE=$(du -sh "${DOCKER_BACKUP_DIR}/${VOLUME_NAME}.tar.gz" 2>/dev/null | cut -f1)
                log "INFO" "  Volume ${RESOLVED_VOLUME} backed up (${VOL_SIZE})"
            else
                log "ERROR" "  Failed to dump volume: ${RESOLVED_VOLUME}"
                VOLUME_ERRORS=$((VOLUME_ERRORS + 1))
            fi
        else
            log "WARN" "  Volume ${VOLUME_NAME} not found -- skipping"
        fi
    done

    if [[ ${VOLUME_ERRORS} -eq 0 ]]; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
    fi
else
    log "WARN" "Docker not available -- skipping volume backups"
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# 4. Backup configuration files (.env, docker-compose.prod.yml)
# ---------------------------------------------------------------------------
TOTAL=$((TOTAL + 1))
CONFIG_DIR="${BACKUP_DIR}/config"
mkdir -p "${CONFIG_DIR}"
CONFIG_ERRORS=0

for CONFIG_FILE in .env docker-compose.prod.yml; do
    SRC="${PROJECT_DIR}/${CONFIG_FILE}"
    if [[ -f "${SRC}" ]]; then
        if cp "${SRC}" "${CONFIG_DIR}/${CONFIG_FILE}" 2>/dev/null; then
            log "INFO" "Backed up config: ${CONFIG_FILE}"
        else
            log "ERROR" "Failed to copy config: ${CONFIG_FILE}"
            CONFIG_ERRORS=$((CONFIG_ERRORS + 1))
        fi
    else
        log "WARN" "Config file not found: ${SRC}"
    fi
done

if [[ ${CONFIG_ERRORS} -eq 0 ]]; then
    SUCCESS=$((SUCCESS + 1))
else
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# 5. Create a manifest with checksums
# ---------------------------------------------------------------------------
log "INFO" "Generating backup manifest..."
MANIFEST="${BACKUP_DIR}/manifest.txt"
{
    echo "Midas Backup Manifest"
    echo "Created: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Source:  ${PROJECT_DIR}"
    echo "---"
    find "${BACKUP_DIR}" -type f ! -name "manifest.txt" -exec sha256sum {} \;
} > "${MANIFEST}" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6. Calculate and display total backup size
# ---------------------------------------------------------------------------
TOTAL_SIZE=$(du -sh "${BACKUP_DIR}" 2>/dev/null | cut -f1)
log "INFO" "------------------------------------------"
log "INFO" "Backup size: ${TOTAL_SIZE}"
log "INFO" "Location:    ${BACKUP_DIR}"

# ---------------------------------------------------------------------------
# 7. Rolling retention -- delete backups older than RETENTION_DAYS
# ---------------------------------------------------------------------------
log "INFO" "Enforcing retention policy (${RETENTION_DAYS} days)..."
DELETED=0
if [[ -d "${DESTINATION}" ]]; then
    while IFS= read -r OLD_BACKUP; do
        if [[ -n "${OLD_BACKUP}" ]]; then
            log "INFO" "  Removing old backup: $(basename "${OLD_BACKUP}")"
            rm -rf "${OLD_BACKUP}"
            DELETED=$((DELETED + 1))
        fi
    done < <(find "${DESTINATION}" -maxdepth 1 -type d -name "midas_backup_*" -mtime "+${RETENTION_DAYS}" 2>/dev/null)
fi
log "INFO" "Retention cleanup: ${DELETED} old backup(s) removed"

# ---------------------------------------------------------------------------
# 8. Final summary and exit code
# ---------------------------------------------------------------------------
log "INFO" "=========================================="
log "INFO" "Backup Summary: ${SUCCESS}/${TOTAL} components succeeded"

if [[ ${SUCCESS} -eq ${TOTAL} ]]; then
    log "INFO" "Result: SUCCESS"
    log "INFO" "=========================================="
    exit 0
elif [[ ${SUCCESS} -gt 0 ]]; then
    log "WARN" "Result: PARTIAL FAILURE (${FAILED} component(s) failed)"
    log "INFO" "=========================================="
    exit 1
else
    log "CRITICAL" "Result: CRITICAL FAILURE (no components backed up)"
    log "INFO" "=========================================="
    exit 2
fi
