# Older live-viewer stack (`infra/`)

`start.sh`, `launch_ue.sh`, `frame_server.py`, `control_bridge.py`, `walk_agent.py`

Built in September for a different demo (the `render_test` third-person project) to **watch**
Unreal live from a laptop over SSH. **The flashlight experiment doesn't use any of it.**

| Script | What it does |
|---|---|
| `start.sh <project.uproject>` | Starts Pixel Streaming's signalling server (used only as an input channel), Unreal in game mode via `launch_ue.sh` writing every frame with `-dumpmovie`, and `frame_server.py` |
| `launch_ue.sh` | Unreal in `-game` mode, headless |
| `frame_server.py` | Serves the dumped frames as an MJPEG stream on one port (`ssh -L 9847:localhost:8090 <you>@<server>`, then open `http://localhost:9847`); its buttons drive the camera through `control_bridge.py` |
| `control_bridge.py` | A hidden Chromium that connects to Pixel Streaming and injects keyboard/mouse input |
| `walk_agent.py` | Walks the character in patterns as a smoke test |

They use the system `python3` plus `requests` and `websocket-client` (not in `requirements.txt`),
and `launch_ue.sh` still passes `-gpuindex`, which Unreal ignores (see
[start_editor.md](start_editor.md)). `stop.sh` also stops these processes.
