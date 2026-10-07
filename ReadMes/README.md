# ReadMes

How to set up, reproduce and adapt the experiments in this repo. Start here, then open the
README for the script you want to run or change.

- **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**: how the pieces fit together (Unreal ↔ Python), with
  diagrams, the coordinate system and a glossary.
- **[../reports/meeting3_report_notes.md](../reports/meeting3_report_notes.md)**: *why* things are
  the way they are: decisions, measurements and results of the Oct 6 experiment.

## The scripts

| Script | What it does | README |
|---|---|---|
| `infra/start_editor.sh`, `infra/stop.sh` | Start headless Unreal (on a chosen GPU, with the command server) and stop it | [start_editor.md](start_editor.md) |
| `unreal_scripts/command_server.py` | Runs **inside** Unreal; accepts named commands (spawn, light, capture, mask, ...) over a local port | [command_server.md](command_server.md) |
| `bridge_client/unreal_client.py` | Python client for the server; library and quick command-line checks | [unreal_client.md](unreal_client.md) |
| `training/scene.py` | Camera/light geometry, the `Scene` wrapper, and the experiment commands: `sweep`, `check-objects`, `calibrate`, `map` | [scene.md](scene.md) |
| `training/vision.py` | The vision models: YOLO9000 scorer (verdict, top-3, box check) and YOLO11m | [vision.md](vision.md) |
| `training/deliverables.py` | Builds the figures in `deliverables/` from a sweep | [deliverables.md](deliverables.md) |
| `training/*.json`, `scene_config.json`, `calibration.md` | Ground-truth classes, model scale fixes, camera/light settings | [data_files.md](data_files.md) |
| `infra/start.sh`, `launch_ue.sh`, `frame_server.py`, `control_bridge.py`, `walk_agent.py` | Older live-viewer stack (not used by the flashlight experiment) | [legacy_live_viewer.md](legacy_live_viewer.md) |

## Setup (on the lab server)

1. **Get the code** (your own copy, so your edits don't affect anyone else's):
   ```bash
   git clone https://github.com/alek03/lighting-VR-RL.git ~/vr-nav-rl
   cd ~/vr-nav-rl
   ```
2. **Python environment.** `python3 -m venv` does not work on this host (no `ensurepip`), so use
   `virtualenv`; torch must come from the CUDA 12.8 index (the driver supports CUDA ≤ 12.8):
   ```bash
   virtualenv .venv
   .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   .venv/bin/pip install -r requirements.txt
   ```
3. **YOLO9000 (Darknet)** is already built at `/opt/Unrealprojects/darknet` (weights, class
   tree, Python wrapper). Nothing to install; set `DARKNET_DIR` if you use another copy.
4. **Your own copy of the Unreal project.** Unreal writes into the project folder (`Saved/`,
   `Intermediate/`), so you need a copy you own. To reproduce the Oct 6 experiment exactly, copy
   the frozen version:
   ```bash
   mkdir /opt/Unrealprojects/tufaelz_frozen_2026-10-06_$USER
   rsync -a /opt/Unrealprojects/tufaelz_frozen_2026-10-06/{Content,Config,Scripts,Docs,tufaelz.uproject} /opt/Unrealprojects/tufaelz_frozen_2026-10-06_$USER/
   ```
   (For the live project's current state, copy from `/opt/Unrealprojects/tufaelz` the same way.)
5. **Pick a GPU and a port nobody else is using:**
   ```bash
   nvidia-smi                      # GPU numbers 0-3 and what's running on them
   ss -tln | grep 678              # command-server ports in use (6780, 6781, ...)
   ```

## Reproduce the Oct 6 experiment

Replace `<copy>`, `<gpu>` and `<port>` with yours.

```bash
cd ~/vr-nav-rl
export UE_COMMAND_PORT=<port>                                        # e.g. 6782
export VRNAV_POSE=<copy>/Scripts/pose.json                           # camera pose from your copy
PROJECT=<copy>/tufaelz.uproject GPU_INDEX=<gpu> ./infra/start_editor.sh    # ~1-2 min (11 min first time)
.venv/bin/python -m training.scene check-objects                     # ~5 min: every object framed? sizes OK?
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
  .venv/bin/python -m training.scene sweep --out captures/sweep_mine # ~60 min, resumable
.venv/bin/python -m training.deliverables captures/sweep_mine --out deliverables_mine   # ~5 min
./infra/stop.sh                                                      # stops only your own Unreal
```

Settings that reproduce Oct 6 are the defaults: `training/scene_config.json` (camera 165 cm,
object fills 50% of a 640×640 frame, light 120 cm away, 0 settle frames), a 12 × 5 light grid
(every 30°, heights 5/25/45/65/85°), `training/object_classes.json` (107 objects) and
`training/object_scale.json` (7 model fixes). The Oct 6 results are in the meeting notes.

## Adapting it

| To change | Where |
|---|---|
| GPU, port, Unreal project | `GPU_INDEX`, `UE_COMMAND_PORT`, `PROJECT` (environment, see [start_editor.md](start_editor.md)) |
| Camera pose file, Darknet location | `VRNAV_POSE`, `DARKNET_DIR` (environment) |
| Light grid (directions, heights) | `sweep --az-step 15 --elevations 10 30 50 70` ([scene.md](scene.md)) |
| Which objects | `sweep --objects Mug Chair ...`, or edit `object_classes.json` ([data_files.md](data_files.md)) |
| Camera height, framing, light radius | `training/scene_config.json`, or `sweep --camera-height` ([data_files.md](data_files.md)) |
| What counts as a correct answer | `object_classes.json` (classes) and `Yolo9000.score` (box rule) ([vision.md](vision.md)) |
| A different vision model | add a scorer to `vision.py` and use it in `scene.frame_row` ([vision.md](vision.md)) |
| A new Unreal command | add a function to `COMMANDS` in `command_server.py` ([command_server.md](command_server.md)) |
| Room geometry in the figures | `TABLE`, `OBJ`, `RADIUS` at the top of `training/deliverables.py` |

## Common problems

| Symptom | Cause / fix |
|---|---|
| `something is already listening on 6780` | Another editor uses that port. Pick a free one (`ss -tln`). |
| Editor on the wrong GPU | `start_editor.sh` warns about it. Unreal ignores `-gpuindex`; the script maps `GPU_INDEX` to `-graphicsadapter` ([start_editor.md](start_editor.md)). |
| `Map load failed` / "found 0 BP_TableObjectSpawner" | The map didn't load. Pass the map as a file path (the script does this by default). |
| Objects render black | The flashlight isn't on lighting channel 1. Already fixed in `command_server.py`; check you use this repo's version. |
| Permission denied in the project folder | You're running someone else's copy. Make your own (Setup step 4). |
| An object overflows the frame or is tiny | Its model's drawn size ≠ its bounds. Run `check-objects`; add the suggested scale to `object_scale.json`. |
