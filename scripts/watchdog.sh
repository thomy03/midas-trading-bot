#!/bin/bash
# Watchdog script for tradingbot-agent
# Checks if container is running and restarts if down

CONTAINER="tradingbot-agent"
LOGFILE="/root/tradingbot-github/logs/watchdog.log"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "$(timestamp) | WARN | ${CONTAINER} is DOWN - restarting..." >> "$LOGFILE"
    docker start "$CONTAINER"
    if [ $? -eq 0 ]; then
        echo "$(timestamp) | INFO | ${CONTAINER} restarted successfully" >> "$LOGFILE"
    else
        echo "$(timestamp) | ERROR | Failed to restart ${CONTAINER}" >> "$LOGFILE"
    fi
else
    echo "$(timestamp) | OK | ${CONTAINER} is running" >> "$LOGFILE"
fi
