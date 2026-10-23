#!/usr/bin/env bash
# Brings up the full stack: Unreal headless, plus the live viewer.
#
#   ./infra/start.sh /opt/Unrealprojects/render_test/render_test.uproject
#
# Three processes, because each does one thing:
#   signalling server  - Pixel Streaming's, used ONLY as the input channel;
#                        the camera buttons and walk_agent need it
#   Unreal             - -dumpmovie feeds the viewer, -PixelStreamingURL the input
#   frame_server       - serves frames as MJPEG on one TCP port, loopback only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PS_ROOT="${PS_ROOT:-$HOME/ps58}"
STREAMER_PORT="${STREAMER_PORT:-8888}"
PLAYER_PORT="${PLAYER_PORT:-8080}"
VIEW_PORT="${VIEW_PORT:-8090}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
SSH_TARGET="${UE_SSH_TARGET:-<user>@<host>}"

PROJECT="${1:?usage: start.sh <project.uproject>}"
shift || true

mkdir -p "$LOG_DIR"

if ss -tln 2>/dev/null | grep -q ":${STREAMER_PORT}"; then
    echo "signalling server already running"
else
    (cd "${PS_ROOT}/SignallingWebServer" && nohup node ./dist/index.js --serve \
        --http_root "${PS_ROOT}/SignallingWebServer/www" \
        --player_port "${PLAYER_PORT}" --streamer_port "${STREAMER_PORT}" \
        --log_folder "${LOG_DIR}/signalling" \
        > "${LOG_DIR}/signalling.log" 2>&1 &)
    sleep 5
    echo "signalling server up"
fi

nohup "${REPO_ROOT}/infra/launch_ue.sh" "$PROJECT" \
    -dumpmovie \
    -PixelStreamingURL="ws://127.0.0.1:${STREAMER_PORT}" \
    "$@" > "${LOG_DIR}/unreal.log" 2>&1 &

echo "Unreal starting (map load takes ~30s)"

PYTHONUNBUFFERED=1 nohup python3 "${REPO_ROOT}/infra/frame_server.py" \
    --port "${VIEW_PORT}" > "${LOG_DIR}/frame_server.log" 2>&1 &

cat <<EOF

Viewer starting on 127.0.0.1:${VIEW_PORT}. Camera buttons need the control
bridge, which takes ~15s more to negotiate WebRTC.

From your own machine (any free local port on the left):
  ssh -L 9847:localhost:${VIEW_PORT} ${SSH_TARGET}
  then open http://localhost:9847

Stop everything:  ./infra/stop.sh
EOF
