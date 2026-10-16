#!/usr/bin/env bash
# Start Unreal headless with server.py loaded, using the settings in config.json.
# Returns when the server is ready (about 1 min; the first run of a project ~10 min).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
LOG="$REPO/logs/editor.log"
mkdir -p "$REPO/logs"

read -r ENGINE PROJECT MAP GPU PORT < <(python3 -c '
import json, os
c = json.load(open("config.json")); u = c["unreal"]
print(u["engine"], u["project"], os.path.join(os.path.dirname(u["project"]), u["map"]), c["gpu"], u["port"])')

for _ in $(seq 30); do                       # a just-stopped Unreal can hold the port a moment
    ss -tln | grep -q "127.0.0.1:$PORT " || break
    sleep 1
done
if ss -tln | grep -q "127.0.0.1:$PORT "; then
    echo "port $PORT is already in use; is Unreal already running?"; exit 1
fi

# Files Unreal creates stay writable by labmates who run the same project.
umask 000
# Unreal ignores -gpuindex. -graphicsadapter counts GPUs in Vulkan's order, which on this
# server is nvidia-smi's shifted by one (adapter 0 = GPU 3, 1 = GPU 0, 2 = GPU 1).
ADAPTER=$(( (GPU + 1) % $(nvidia-smi -L | wc -l) ))
# -RenderOffScreen: no display here. TextureStreaming=0: otherwise textures stay blurry,
# because Unreal picks texture detail from the (unused) main viewport, not our camera.
# -ExecCmds, not -ExecutePythonScript, which would quit Unreal after the script runs.
export LIGHTING_RL_PORT="$PORT"
nohup "$ENGINE/Engine/Binaries/Linux/UnrealEditor" "$PROJECT" "$MAP" \
    -RenderOffScreen -vulkan -graphicsadapter="$ADAPTER" -unattended -nosplash -nosound -log \
    -dpcvars="r.TextureStreaming=0" \
    -ExecCmds="py $REPO/unreal_bridge/server.py" > "$LOG" 2>&1 &
PID=$!
echo "Unreal starting on GPU $GPU (log: $LOG)"

for _ in $(seq 360); do
    if grep -q "server: listening" "$LOG"; then
        echo "ready on port $PORT"
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | grep -qx "$PID" \
            || echo "WARNING: Unreal is not on GPU $GPU; check nvidia-smi"
        exit 0
    fi
    kill -0 "$PID" 2>/dev/null || { echo "Unreal exited during startup; see $LOG"; exit 1; }
    sleep 5
done
echo "still not ready after 30 min; see $LOG"; exit 1
