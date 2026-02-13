#!/usr/bin/env bash
# =============================================================================
# Midas Trading Bot - Restore Script
# =============================================================================
# Restores a previously created backup to the project directory.
#
# Usage:
#   ./scripts/restore.sh --from /backups/midas/midas_backup_20260214_030000
#   ./scripts/restore.sh --from /backups/midas/midas_backup_20260214_030000 --dry-run
#
# Options:
#   --from DIR      Path to the backup directory (required)
#   --dry-run       List what would be restored without making changes
#   --skip-confirm  Skip the interactive confirmation prompt
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BACKUP_DIR=""
DRY_RUN=false
SKIP_CONFIRM=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-confirm)
            SKIP_CONFIRM=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --from /path/to/backup_dir [--dry-run] [--skip-confirm]"
            echo ""
            echo "Options:"
            echo "  --from DIR       Path to backup directory (required)"
            echo "  --dry-run        List contents without restoring"
            echo "  --skip-confirm   Skip interactive confirmation prompt"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
}

# ---------------------------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------------------------
if [[ -z "${BACKUP_DIR}" ]]; then
    echo "Error: --from argument is required." >&2
    echo "Usage: $0 --from /path/to/backup_dir [--dry-run]" >&2
    exit 1
fi

if [[ ! -d "${BACKUP_DIR}" ]]; then
    log "ERROR" "Backup directory does not exist: ${BACKUP_DIR}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate backup structure
# ---------------------------------------------------------------------------
log "INFO" "Validating backup: ${BACKUP_DIR}"

EXPECTED_FILES=()
FOUND_FILES=()
MISSING_FILES=()

for ITEM in data.tar.gz models.tar.gz config; do
    EXPECTED_FILES+=("${ITEM}")
    if [[ -e "${BACKUP_DIR}/${ITEM}" ]]; then
        FOUND_FILES+=("${ITEM}")
    else
        MISSING_FILES+=("${ITEM}")
    fi
done

# Check for optional docker volume backups
HAS_DOCKER_VOLUMES=false
if [[ -d "${BACKUP_DIR}/docker_volumes" ]]; then
    HAS_DOCKER_VOLUMES=true
fi

if [[ ${#FOUND_FILES[@]} -eq 0 ]]; then
    log "ERROR" "No recognizable backup components found in ${BACKUP_DIR}"
    log "ERROR" "Expected at least one of: data.tar.gz, models.tar.gz, config/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Display backup contents
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Midas Backup Restore"
echo "============================================"
echo "  Backup source : ${BACKUP_DIR}"
echo "  Restore target: ${PROJECT_DIR}"
echo ""

echo "  Available components:"
for F in "${FOUND_FILES[@]}"; do
    case "${F}" in
        data.tar.gz)
            SIZE=$(du -sh "${BACKUP_DIR}/data.tar.gz" 2>/dev/null | cut -f1)
            echo "    [+] data.tar.gz (${SIZE}) -> data/"
            ;;
        models.tar.gz)
            SIZE=$(du -sh "${BACKUP_DIR}/models.tar.gz" 2>/dev/null | cut -f1)
            echo "    [+] models.tar.gz (${SIZE}) -> models/"
            ;;
        config)
            echo "    [+] config/ -> .env, docker-compose.prod.yml"
            ls -1 "${BACKUP_DIR}/config/" 2>/dev/null | while read -r CF; do
                echo "        - ${CF}"
            done
            ;;
    esac
done

if [[ "${HAS_DOCKER_VOLUMES}" == true ]]; then
    echo "    [+] docker_volumes/"
    ls -1 "${BACKUP_DIR}/docker_volumes/" 2>/dev/null | while read -r VF; do
        SIZE=$(du -sh "${BACKUP_DIR}/docker_volumes/${VF}" 2>/dev/null | cut -f1)
        echo "        - ${VF} (${SIZE})"
    done
fi

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo ""
    echo "  Missing components (will be skipped):"
    for M in "${MISSING_FILES[@]}"; do
        echo "    [-] ${M}"
    done
fi

# Check manifest
if [[ -f "${BACKUP_DIR}/manifest.txt" ]]; then
    echo ""
    echo "  Manifest found. Backup created:"
    grep "^Created:" "${BACKUP_DIR}/manifest.txt" 2>/dev/null | sed 's/^/    /'
fi

echo ""
echo "============================================"

# ---------------------------------------------------------------------------
# Dry-run: stop here
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    log "INFO" "Dry run complete. No changes were made."
    echo ""
    echo "To perform the restore, run:"
    echo "  $0 --from ${BACKUP_DIR}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Confirmation prompt
# ---------------------------------------------------------------------------
if [[ "${SKIP_CONFIRM}" != true ]]; then
    echo ""
    echo "WARNING: This will overwrite existing data in:"
    echo "  ${PROJECT_DIR}/data/"
    echo "  ${PROJECT_DIR}/models/"
    echo "  ${PROJECT_DIR}/.env"
    echo "  ${PROJECT_DIR}/docker-compose.prod.yml"
    echo ""
    read -r -p "Continue with restore? (yes/no): " CONFIRM
    if [[ "${CONFIRM}" != "yes" ]]; then
        log "INFO" "Restore cancelled by user."
        exit 0
    fi
fi

echo ""
log "INFO" "Starting restore..."

# ---------------------------------------------------------------------------
# Restore data/
# ---------------------------------------------------------------------------
if [[ -f "${BACKUP_DIR}/data.tar.gz" ]]; then
    log "INFO" "Restoring data/ directory..."
    # Create a pre-restore backup of current data
    if [[ -d "${PROJECT_DIR}/data" ]]; then
        PRE_BACKUP="${PROJECT_DIR}/data.pre-restore.$(date '+%Y%m%d_%H%M%S')"
        log "INFO" "  Saving current data/ to ${PRE_BACKUP}"
        mv "${PROJECT_DIR}/data" "${PRE_BACKUP}"
    fi
    mkdir -p "${PROJECT_DIR}/data"
    if tar -xzf "${BACKUP_DIR}/data.tar.gz" -C "${PROJECT_DIR}" 2>/dev/null; then
        log "INFO" "  data/ restored successfully"
    else
        log "ERROR" "  Failed to restore data/"
        # Attempt to restore from pre-backup
        if [[ -d "${PRE_BACKUP}" ]]; then
            log "WARN" "  Rolling back data/ to pre-restore state"
            rm -rf "${PROJECT_DIR}/data"
            mv "${PRE_BACKUP}" "${PROJECT_DIR}/data"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Restore models/
# ---------------------------------------------------------------------------
if [[ -f "${BACKUP_DIR}/models.tar.gz" ]]; then
    log "INFO" "Restoring models/ directory..."
    if [[ -d "${PROJECT_DIR}/models" ]]; then
        PRE_BACKUP="${PROJECT_DIR}/models.pre-restore.$(date '+%Y%m%d_%H%M%S')"
        log "INFO" "  Saving current models/ to ${PRE_BACKUP}"
        mv "${PROJECT_DIR}/models" "${PRE_BACKUP}"
    fi
    mkdir -p "${PROJECT_DIR}/models"
    if tar -xzf "${BACKUP_DIR}/models.tar.gz" -C "${PROJECT_DIR}" 2>/dev/null; then
        log "INFO" "  models/ restored successfully"
    else
        log "ERROR" "  Failed to restore models/"
        if [[ -d "${PRE_BACKUP}" ]]; then
            log "WARN" "  Rolling back models/ to pre-restore state"
            rm -rf "${PROJECT_DIR}/models"
            mv "${PRE_BACKUP}" "${PROJECT_DIR}/models"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Restore config files
# ---------------------------------------------------------------------------
if [[ -d "${BACKUP_DIR}/config" ]]; then
    log "INFO" "Restoring config files..."
    for CONFIG_FILE in .env docker-compose.prod.yml; do
        SRC="${BACKUP_DIR}/config/${CONFIG_FILE}"
        DST="${PROJECT_DIR}/${CONFIG_FILE}"
        if [[ -f "${SRC}" ]]; then
            # Back up current config before overwriting
            if [[ -f "${DST}" ]]; then
                cp "${DST}" "${DST}.pre-restore.$(date '+%Y%m%d_%H%M%S')"
            fi
            cp "${SRC}" "${DST}"
            log "INFO" "  Restored: ${CONFIG_FILE}"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Restore Docker volumes (if docker available and backups exist)
# ---------------------------------------------------------------------------
if [[ "${HAS_DOCKER_VOLUMES}" == true ]] && command -v docker &>/dev/null; then
    log "INFO" "Restoring Docker volumes..."
    for VOLUME_ARCHIVE in "${BACKUP_DIR}"/docker_volumes/*.tar.gz; do
        [[ -f "${VOLUME_ARCHIVE}" ]] || continue
        VOLUME_NAME="$(basename "${VOLUME_ARCHIVE}" .tar.gz)"

        # Try to find the actual volume name (with or without project prefix)
        RESOLVED_VOLUME=""
        for CANDIDATE in "${VOLUME_NAME}" "midas_${VOLUME_NAME}" "midas-${VOLUME_NAME}"; do
            if docker volume inspect "${CANDIDATE}" &>/dev/null; then
                RESOLVED_VOLUME="${CANDIDATE}"
                break
            fi
        done

        if [[ -z "${RESOLVED_VOLUME}" ]]; then
            log "WARN" "  Volume ${VOLUME_NAME} does not exist -- creating it"
            docker volume create "${VOLUME_NAME}" &>/dev/null || true
            RESOLVED_VOLUME="${VOLUME_NAME}"
        fi

        log "INFO" "  Restoring volume: ${RESOLVED_VOLUME}"
        if docker run --rm \
            -v "${RESOLVED_VOLUME}:/volume" \
            -v "$(dirname "${VOLUME_ARCHIVE}"):/backup:ro" \
            alpine \
            sh -c "rm -rf /volume/* /volume/..?* /volume/.[!.]* 2>/dev/null; tar -xzf /backup/$(basename "${VOLUME_ARCHIVE}") -C /volume" 2>/dev/null; then
            log "INFO" "  Volume ${RESOLVED_VOLUME} restored"
        else
            log "ERROR" "  Failed to restore volume: ${RESOLVED_VOLUME}"
        fi
    done
elif [[ "${HAS_DOCKER_VOLUMES}" == true ]]; then
    log "WARN" "Docker volume backups found but docker is not available -- skipping"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
log "INFO" "============================================"
log "INFO" "Restore complete."
log "INFO" "============================================"
echo ""
echo "Next steps:"
echo "  1. Review restored files in ${PROJECT_DIR}"
echo "  2. Restart services:  docker compose -f docker-compose.prod.yml up -d"
echo "  3. Verify health:     docker compose -f docker-compose.prod.yml ps"
echo ""
