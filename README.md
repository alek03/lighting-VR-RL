# lighting-rl

A camera looks at an object on a table in the tufaelz Unreal project. A flashlight moves on a dome
around the object, and a vision model (YOLOE-26) tries to identify the object. The goal is an RL
agent that finds the light position where the object is identified best.

Three parts:

| Folder | What it does |
|---|---|
| `unreal_bridge/` | Start Unreal on this server and control it from Python: objects, camera, zoom, flashlight, pictures |
| `vision/` | YOLOE-26 scoring, the light-position sweep, and figures |
| `rl/` | The RL agent (to do) |

All settings are in `config.json`: camera position/angle/zoom, light distance and grid, picture
size, GPU, model, project path.

## Setup

```
cd /home/axm6766/lighting-rl
.venv/bin/python -c "import ultralytics, torch; print(torch.cuda.is_available())"   # True
```

`.venv` currently points to `~/vr-nav-rl/.venv`. For a fresh environment: create one with
`virtualenv .venv`, install torch from the CUDA 12.8 index (see `requirements.txt`), then
`.venv/bin/pip install -r requirements.txt`. Point Ultralytics at `models/` once, so it finds the text
model there: `.venv/bin/yolo settings weights_dir=$PWD/models`.

## Unreal

```
./unreal_bridge/start.sh     # ~1 min; returns when ready (first run of a project: ~10 min)
./unreal_bridge/stop.sh      # stop it when done; it holds a GPU
```

Run commands from the repo folder with `.venv/bin/python -m ...`, so the folders can import each other.

```python
from unreal_bridge import Unreal

with Unreal() as ue:                          # on exit, removes everything it added
    ue.objects()                              # the 115 object names
    ue.spawn("Mug")                           # put an object on the table
    ue.set_light(azimuth=90, elevation=45)    # flashlight on the dome (radius from config)
    image = ue.capture()                      # numpy RGB array, 640 x 640
    box = ue.true_box()                       # [x0, y0, x1, y1] around the object
    ue.set_camera(fov=30)                     # move/turn/zoom; left-out values come from config
    ue.set_light_at([1100, -250, 200], target=[1120, -250, 87])   # light at any exact point
```

Light angles are relative to the camera: azimuth 0 = the camera's side of the object, 90 = the
camera's right, 180 = behind; elevation 0 = level with the object, 90 = straight above.

`server.py` runs inside Unreal and takes these commands from Python. You shouldn't need to touch it.

## Vision

```python
from vision.yoloe import Detector

det = Detector()                                   # all object names are the prompts
result = det.score(image, "Mug", true_box=box)     # verdict, confidence, top 3 guesses, box, iou
```

A picture is `correct` if YOLOE's top guess is the object's name and its box is on the object
(IoU or object fraction at least `vision.on_object`).

Sweep: a picture at each of the 60 light positions (12 directions x 5 heights) for every object,
then scores and figures:

```
.venv/bin/python -m vision.sweep render --out outputs/sweep_2026-10-17     # needs Unreal, ~1 h for all objects
.venv/bin/python -m vision.sweep score  --out outputs/sweep_2026-10-17     # no Unreal, ~4 min
.venv/bin/python -m vision.deliverables outputs/sweep_2026-10-17           # figures -> <sweep>/deliverables
```

`sweep.csv` has one row per picture: object, light position, verdict, confidence, top 3 guesses,
YOLOE's box and the true box. Changing the model only needs `score` again, not new pictures.

## Things to know

- **Project copy.** `config.json` points at a copy of the tufaelz project in
  `~/tufaelz_copy` (Config + Content, copied 2026-10-16). Several people running Unreal in the
  shared `/opt/Unrealprojects/tufaelz` leave files the others can't write to, and Unreal crashes on
  them. The copy also keeps labmates' project changes from shifting an experiment. To pick up a
  newer project: copy `tufaelz.uproject`, `Config` and `Content` again.
- **Thimble** is broken in the project: drawn ~5x too large and invisible to the mask render.
  `spawn` raises an error for it and the sweep skips it.
- **First use of an object** in a project copy: Unreal takes a moment to prepare it; `spawn` waits
  until it is drawn.
- **GPU numbers** are nvidia-smi's. Unreal counts GPUs differently; `start.sh` handles that and
  warns if Unreal lands on the wrong GPU.
