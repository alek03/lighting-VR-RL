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
