#!/usr/bin/env bash
# Stops everything start.sh brought up, and reports the GPUs so you can confirm
# nothing is left holding VRAM on this shared machine.
#
# Matches on `ps` output rather than using `pkill -f`: pkill's pattern also
# matches the shell running it, so it kills itself before reaching the target.
# Only your own processes: labmates run editors on this machine too.
set -uo pipefail

stop() {
    local label="$1" pattern="$2"
    local pids
    pids=$(ps -u "$(id -u)" -o pid,args | grep -- "$pattern" | grep -v grep | awk '{print $1}')
    if [ -z "$pids" ]; then
        echo "${label}: not running"
        return
    fi
    kill $pids 2>/dev/null
    sleep 4
    for p in $pids; do
        ps -p "$p" >/dev/null 2>&1 && kill -9 "$p" 2>/dev/null
    done
    echo "${label}: stopped"
}

stop "unreal"       "Binaries/Linux/UnrealEditor"
stop "frame server" "frame_server.py"
stop "control bridge" "remote-debugging-port=923"
# Started after a cd, so its args are the relative "node ./dist/index.js".
stop "signalling"   "dist/index.js"

echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
