#!/usr/bin/env bash
# Stop your own Unreal (not labmates'), then show GPU use.
PATTERN="Binaries/Linux/UnrealEditor"
if pkill -u "$(id -u)" -f "$PATTERN"; then
    for _ in $(seq 30); do                   # Unreal takes a few seconds to shut down
        pgrep -u "$(id -u)" -f "$PATTERN" >/dev/null || break
        sleep 1
    done
    pkill -9 -u "$(id -u)" -f "$PATTERN" && echo "Unreal force-stopped" || echo "Unreal stopped"
else
    echo "Unreal was not running"
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
