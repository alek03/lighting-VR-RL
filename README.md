# lighting-VR-RL

RL on a VR scene in Unreal Engine 5.8, running headless on a shared GPU server. First task: a fixed
camera faces a table, an object is spawned on it, and an agent moves a flashlight on a sphere
around the object to maximise a vision model's confidence in identifying it.

## Layout

- `unreal_scripts/command_server.py` — runs *inside* the Unreal editor; localhost command server
- `bridge_client/unreal_client.py` — Python client for it, library and CLI
- `training/scene.py` — camera/light geometry, object masks, calibration and the lighting sweep
- `training/vision.py` — YOLO9000 scorer (Darknet, per-class probabilities) and YOLO11m scorer
- `training/object_classes.json` — which YOLO9000 classes count as correct for each object
- `training/object_scale.json` — scale fixes for models whose drawn size ≠ stored bounds
- `infra/` — starting/stopping the editor, plus the older live-viewer stack (see below)
- `reports/` — running report notes; `HOW_IT_WORKS.md` — guided tour of the code
- `captures/`, `logs/`, `deliverables/` — run artifacts, gitignored

The Unreal project is not in this repo: `/opt/Unrealprojects/tufaelz` (mostly binary `.uasset`
content, which git handles poorly). Its own `Scripts/`, `Docs/` and `Content/Python/` hold the
object spawner bridge this repo builds on. Sweeps run on a frozen copy,
`/opt/Unrealprojects/tufaelz_frozen_2026-10-06`, because the live project is edited by others.

## Setup

    virtualenv .venv
    .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    .venv/bin/pip install -r requirements.txt

`python3 -m venv` does not work on this host (`ensurepip` is missing), and torch must come from the
cu128 index: the driver supports CUDA 12.8 at most.

## Usage

    export UE_COMMAND_PORT=6781     # a labmate's editor uses 6780
    PROJECT=/opt/Unrealprojects/tufaelz_frozen_2026-10-06/tufaelz.uproject ./infra/start_editor.sh   # GPU 1 by default
    .venv/bin/python -m bridge_client.unreal_client check --object Mug     # spawn, capture, clean up
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        .venv/bin/python -m training.scene sweep --out captures/sweep_2026-10-06   # ~70 min, resumable
    ./infra/stop.sh                 # stops only your own Unreal processes

Commands the server accepts: `ping`, `describe`, `status`, `spawn`, `remove`, `scale_object`,
`capture_setup`, `capture`, `capture_mask`, `capture_cleanup`, `light_setup`, `set_light`,
`light_cleanup`, `play`, `set_cvars`. It never executes arbitrary code.

## Host constraints

These shaped every decision here and are worth knowing before changing anything.

**No display.** `-RenderOffScreen` is required: Vulkan can render on the GPU but cannot present to a
surface. Xvfb does *not* help — it is software-only, and the engine exits on "Cannot find a
compatible Vulkan device that supports surface presentation".

**Epic's Python remote execution cannot be discovered on Linux.** Its discovery is UDP multicast,
and with `RemoteExecutionMulticastBindAddress=127.0.0.1` a Linux socket never receives the packets
(Windows delivers them anyway). Binding to `0.0.0.0` would fix discovery but expose arbitrary Python
execution to the network, hence `command_server.py`: localhost TCP, named commands only.

**`-ExecutePythonScript` quits the editor** after the script runs. The server is loaded with
`-ExecCmds="py <script>"` instead.

**Only port 22 is reachable.** WebRTC/Pixel Streaming media is UDP on dynamic ports and cannot
traverse an SSH tunnel (it stalls at `IceConnectionChecking`). Live viewing therefore uses MJPEG over
one tunnelled TCP port.

**Unreal ignores `-gpuindex`.** Use `-graphicsadapter`, which counts GPUs in Vulkan's order: on
this host `nvidia-smi`'s order shifted by one (adapter 0 = GPU 3, 1 = GPU 0, 2 = GPU 1).
`start_editor.sh` converts and then verifies the placement with `nvidia-smi`.

**Pass the map as a file path.** A `/Game/...` map path fails to load when the project folder
and `.uproject` names differ (as in copies of the project).

**No project authoring.** Creating a project properly needs the editor's GUI wizard; copying a
template directory by hand silently omits `/Game/LevelPrototyping`, `/Game/Characters` and
`/Game/Input`. Author on a machine with a display; use this host for running and training.

## Status

Working (2026-10-06): headless editor on a chosen GPU; all 107 scorable objects spawn, lit by a
flashlight matching the VR torch (incl. its lighting channels), framed per object; object masks
for checking where YOLO's boxes are; YOLO9000 scoring with confidence, top-3 guesses and
correct/partial/wrong verdicts; a resumable 60-position lighting sweep over all objects.

Known issue: 7 models in the Oct 5 project update are drawn at a different size from their
stored bounds (worked around in `training/object_scale.json`; to be reported to the model authors).

Next: deliverables from the sweep (`deliverables/`), then the RL environment.

### Older live-viewer stack

`start.sh`, `launch_ue.sh`, `frame_server.py`, `control_bridge.py` and `walk_agent.py` were built for
the `render_test` ThirdPerson demo: Unreal in `-game` mode, frames via `-dumpmovie`, input injected
through Pixel Streaming and a headless Chromium.
