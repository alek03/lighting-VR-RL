#!/usr/bin/env python3
"""Drive the character around the level so the live view shows motion.

Useful as a smoke test that the whole chain is alive: rendering, frame
extraction, the viewer, and the input path.

Note this is a demo/debug tool, not the RL action channel. Actions ride through
a headless browser and Pixel Streaming's data channel, which is far too much
latency and too many moving parts for a training loop.

    python3 infra/walk_agent.py --pattern circle --duration 300
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_bridge import ControlBridge  # noqa: E402

# (held keys, seconds, turn steps per tick). Movement is camera-relative, so
# walking a circle means holding forward while continuously turning; holding
# w+a alone just tracks a straight diagonal into the nearest wall.
PATTERNS = {
    "circle": [(("w",), 9999, 1)],
    "square": [(("w",), 3, 0), ((), 1, 6)],
    "forwardback": [(("w",), 3, 0), (("s",), 3, 0)],
    "spin": [((), 9999, 2)],
}

TICK = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=sorted(PATTERNS), default="circle")
    ap.add_argument("--duration", type=float, default=120, help="seconds to drive")
    args = ap.parse_args()

    bridge = ControlBridge(debug_port=9231)
    bridge.start()
    print("stream live; driving input")

    steps = PATTERNS[args.pattern]
    deadline = time.time() + args.duration
    held = set()
    try:
        while time.time() < deadline:
            for keys, seconds, turn in steps:
                if time.time() >= deadline:
                    break
                for k in keys:
                    if k not in held:
                        bridge.key(k, True)
                        held.add(k)
                for k in list(held):
                    if k not in keys:
                        bridge.key(k, False)
                        held.discard(k)
                print(f"  {'+'.join(keys) or 'no keys'}, turn={turn}")

                step_end = min(time.time() + seconds, deadline)
                while time.time() < step_end:
                    if turn:
                        bridge.yaw(turn, right=True)
                    time.sleep(TICK)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        for k in list(held):
            bridge.key(k, False)
        bridge.stop()
        print("done")


if __name__ == "__main__":
    main()
