# vision/sweep.py

The light-position sweep: a picture of every object at every light position, then a YOLOE score
for each picture. Two separate steps, so changing the model never means taking new pictures.

```
cd /home/axm6766/lighting-rl
.venv/bin/python -m vision.sweep render --out outputs/sweep_2026-10-17     # needs Unreal running, ~1 h
.venv/bin/python -m vision.sweep score  --out outputs/sweep_2026-10-17     # no Unreal, ~6 min
```

`--objects Mug Banana` renders only those objects (default: all).

## render

For each object: spawn it (with per-object zoom if `camera.fill` is set), save its mask, then put
the light at each of the 60 positions (12 directions x 5 heights from `config.json`) and save a
picture. Writes, inside the `--out` folder:

| File | Contents |
|---|---|
| `frames/<Object>/el45_az090.png` | Picture with the light 45° high, 90° to the camera's right |
| `frames/<Object>/mask.png` | Where the object is (white) |
| `positions.csv` | One row per picture: object, elevation, azimuth, light x/y/z, fov (that object's zoom), frame |
| `settings.json` | The camera, light, image and table settings the sweep was made with |

Objects already in `positions.csv` are skipped, so a stopped run continues where it left off. An
object is only added once all of its pictures exist. Objects that never appear (e.g. Thimble) are
skipped with a message.

## score

Runs YOLOE on every picture in `positions.csv` and writes `sweep.csv`: the same columns plus
`verdict`, `confidence`, `top1`–`top3` with confidences, YOLOE's box, `iou`, `object_fraction`,
and the true box (from `mask.png`). Overwrites any earlier `sweep.csv`; copy it first to keep it.

## Settings used (config.json)

`light.azimuths`, `light.elevations`, and everything the `Unreal` class and `Detector` use.

## Select the tabletop spawner

Run from `/home/mzt5720/vr-nav-rl`. The configured project is
`/opt/Unrealprojects/tufaelz/tufaelz.uproject`. Restart the bridge after updating its code.

```bash
bash unreal_bridge/start.sh

# Original spawner (115 objects)
.venv/bin/python -m vision.sweep render --collection common --out outputs/sweep_common
.venv/bin/python -m vision.sweep score --collection common --out outputs/sweep_common
.venv/bin/python -m vision.deliverables outputs/sweep_common

# New spawner (31 objects from Config/objectlab_objects2.json)
.venv/bin/python -m vision.sweep render --collection new --out outputs/sweep_new
.venv/bin/python -m vision.sweep score --collection new --out outputs/sweep_new
.venv/bin/python -m vision.deliverables outputs/sweep_new
```

Use separate output folders for the two collections. Rendering defaults to `common`.
Scoring can omit `--collection`: it reads the selection and full prompt catalog saved in
`settings.json`. Existing sweeps without a collection are treated as `common`.
Deliverables need no collection option or code changes. Add `--objects Bin Boot` to a new
render command to capture only those objects while keeping all new catalog names as prompts.
The existing ObjectLab controller also supports `objectlab.ps1 --collection common spawn Mug`
and `objectlab.ps1 --collection new spawn Bin` in its Windows PowerShell environment.
