#!/usr/bin/env bash
# Runs the full Unreal editor headless with the tufaelz level open and the
# localhost command server loaded (unreal_scripts/command_server.py).
#
#   ./infra/start_editor.sh            # then talk to it via bridge_client/unreal_client.py
#   PROJECT=/opt/Unrealprojects/tufaelz_frozen_2026-10-06/tufaelz.uproject UE_COMMAND_PORT=6781 ./infra/start_editor.sh
#
# The full editor, not -game, because the project's bridge code uses editor
# subsystems. First launch compiles shaders (~11 min); later launches reuse the cache.
#
# The server is loaded with -ExecCmds="py ..." rather than -ExecutePythonScript,
# which runs the script and then quits the editor.
#
# r.TextureStreaming=0: texture streaming picks resolution from the main viewport, which
# never looks at the table, so our zoomed scene capture otherwise gets blurry low mips.
#
# GPU_INDEX is the nvidia-smi index. Unreal's Vulkan RHI ignores -gpuindex; -graphicsadapter
# works but counts in Vulkan's order, which on this host is nvidia-smi's shifted by one
# (adapter 0 = GPU 3, 1 = GPU 0, 2 = GPU 1). The placement is checked after startup.
#
# The map is passed as a file path: a /Game/... path fails to load when the project folder
# is named differently from the .uproject (as in copies like tufaelz_frozen_*).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UE_ROOT="${UE_ROOT:-/opt/UnrealEngine/UE-5.8.2}"
PROJECT="${PROJECT:-/opt/Unrealprojects/tufaelz/tufaelz.uproject}"
MAP="${MAP:-$(dirname "$PROJECT")/Content/XRFramework/Levels/L_XRTemplate.umap}"
GPU_INDEX="${GPU_INDEX:-1}"
GPU_COUNT="$(nvidia-smi -L | wc -l)"
ADAPTER=$(( (GPU_INDEX + 1) % GPU_COUNT ))
LOG="${REPO_ROOT}/logs/editor.log"
export UE_COMMAND_PORT="${UE_COMMAND_PORT:-6780}"

mkdir -p "${REPO_ROOT}/logs"
if ss -tln 2>/dev/null | grep -q "127.0.0.1:${UE_COMMAND_PORT}"; then
    echo "something is already listening on ${UE_COMMAND_PORT}; is an editor running?"
    exit 1
fi

nohup "${UE_ROOT}/Engine/Binaries/Linux/UnrealEditor" "$PROJECT" "$MAP" \
    -RenderOffScreen -vulkan -graphicsadapter="${ADAPTER}" \
    -unattended -nosplash -nosound -log \
    -dpcvars="r.TextureStreaming=0" \
    -ExecCmds="py ${REPO_ROOT}/unreal_scripts/command_server.py" \
    > "$LOG" 2>&1 &
PID=$!
echo "editor starting (pid $PID, GPU $GPU_INDEX); log: $LOG"

for _ in $(seq 1 360); do
    if grep -q "command_server: listening" "$LOG" 2>/dev/null; then
        echo "ready on 127.0.0.1:${UE_COMMAND_PORT}"
        want="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$GPU_INDEX" '$1==i {print $2}')"
        got="$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v p="$PID" '$1==p {print $2}')"
        if [ "$got" != "$want" ]; then
            echo "WARNING: editor is on GPU bus ${got:-unknown}, not GPU $GPU_INDEX ($want); check nvidia-smi"
        fi
        if grep -q "Map load failed" "$LOG"; then
            echo "WARNING: the map did not load; see $LOG"
        fi
        exit 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "editor exited during startup; see $LOG"
        exit 1
    fi
    sleep 5
done
echo "still starting after 30 min; see $LOG"
exit 1
