#!/usr/bin/env bash
# Launches an Unreal project headless with real GPU rendering.
#
# Usage: ./launch_ue.sh /opt/Unrealprojects/render_test/render_test.uproject [extra UE args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

UE_ROOT="${UE_ROOT:-/opt/UnrealEngine/UE-5.8.2}"
UE_BIN="${UE_ROOT}/Engine/Binaries/Linux/UnrealEditor"
GPU_INDEX="${GPU_INDEX:-3}"
RES_X="${RES_X:-1280}"
RES_Y="${RES_Y:-720}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

PROJECT="${1:?usage: launch_ue.sh <project.uproject> [extra args]}"
shift || true

mkdir -p "$LOG_DIR"

# -RenderOffScreen is load-bearing: this box has no GPU-backed display server, so
# Vulkan cannot present to a surface. Without it the engine dies on a modal
# "Cannot find a compatible Vulkan device that supports surface presentation".
# -gpuindex keeps rendering on one GPU, leaving the rest free for model training.
exec "$UE_BIN" "$PROJECT" \
    -game \
    -vulkan \
    -RenderOffScreen \
    -gpuindex="${GPU_INDEX}" \
    -resx="${RES_X}" -resy="${RES_Y}" \
    -unattended \
    -nosound \
    -nosplash \
    -nopause \
    -log \
    "$@"
