# `training/scene.py`

Everything about placing the camera and the flashlight, plus the experiment commands. All
commands except `sweep --rescore` need the editor running (`start_editor.sh`) and
`UE_COMMAND_PORT` set.

## Commands

### `sweep`: the lighting experiment

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
  .venv/bin/python -m training.scene sweep --out captures/sweep_mine
```

For every object in `object_classes.json` (or `--objects ...`): place it (scaled if listed in
`object_scale.json`), aim and zoom the camera, capture its **mask**, then for every light position
render a frame, score it with YOLO9000 (and YOLO11m if the object has a COCO class), and check
YOLO's box against the mask. About 35 s per object; the Oct 6 run took 59 min.

| Option | Default | Meaning |
|---|---|---|
| `--out` | (required) | Output folder |
| `--objects` | all 107 | Subset of objects |
| `--az-step` | 30 | Degrees between light directions (12 directions) |
| `--elevations` | 5 25 45 65 85 | Light heights in degrees above the object's centre |
| `--camera-height` | from `scene_config.json` (165) | Camera height in cm |
| `--rescore` | | Re-score the saved frames instead of rendering (no Unreal needed; ~12 min) |

**Output:** `<out>/sweep.csv` (one row per object × light position), `<out>/frames/<Object>/el.._az...png`
and `<out>/frames/<Object>/mask.png`. **Resumable:** run the same command again and finished
objects are skipped (a half-finished object is redone).

**`sweep.csv` columns**

| Column(s) | Meaning |
|---|---|
| `object`, `id`, `elevation`, `azimuth` | Which object (catalogue id) and light position |
| `light_x/y/z` | Light position in world cm |
| `frame` | The saved image, relative to `<out>` |
| `accepted` | Classes that count as correct (from `object_classes.json`) |
| `verdict` | `correct` / `partial` / `wrong` / `none` (see [vision.md](vision.md)) |
| `confidence` | YOLO9000's best confidence for an accepted class, anywhere in the frame |
| `top1..3`, `top1..3_conf`, `top1_wnid` | Its three most specific guesses; WordNet ID of the first |
| `box_*`, `objectness` | YOLO's most confident box and how sure it is there's an object |
| `iou`, `object_fraction`, `coverage`, `centre_in_box` | Box check against the mask: overlap with the mask's box; share of YOLO's box that is object pixels; share of the object inside the box; whether the object's centre is inside |
| `truth_*` | The mask's box (the true object box) |
| `yolo11m_confidence` | YOLO11m's confidence, for objects with a COCO class |

**Re-scoring** (after changing `object_classes.json` or the box rule): add `--rescore`. Light
positions and YOLO11m scores are kept; the old table is saved as `sweep_before_rescore.csv`.

### `check-objects`: are all objects framed and sized correctly?

```bash
.venv/bin/python -m training.scene check-objects            # --objects ..., --out captures/object_check
```

Renders each object once and compares its silhouette with its stored size (bounds). Flags objects
that overflow the frame, are missing, or don't match their bounds (IoU < 0.4), and for flagged
objects measures the drawn width and prints a **suggested scale** for `object_scale.json`. Writes
`object_check.json`, `contact_sheet.jpg` (green = silhouette box, magenta = bounds) and one frame
per object. Run it whenever the Unreal objects change. An object that isn't drawn at all at scale 1
(as the Thimble was) gets no suggestion: try a few scales with `bridge_client` + `scale_object`.

### `calibrate` and `map` (first runs)

- `calibrate`: measures settle frames and chooses framing (`fill`) and light radius with YOLO11m
  on the original 10 objects; writes `scene_config.json`, `calibration.md`, `logs/calibration.json`.
  Re-running it overwrites the settings the Oct 6 experiment used.
- `map`: draws `captures/room_map.png` (top and side view of camera, zoom range and light sphere).

## The code, for editing

- **Geometry (no Unreal):** `camera_pose` (camera at the `pose.json` position, aimed at the object,
  zoomed so the object fills `fill` of the frame), `light_position` (azimuth/elevation/radius →
  world point; angles relative to the camera: azimuth 0 = camera's side, 90 = its right;
  elevation 0 = level, 90 = overhead), `projected_box` (an object's 3D bounds → image box),
  `mask_box`.
- **`Scene`**: a wrapper over the commands: `load(name)`, `light(az, el, r)`, `capture(settle)`,
  `capture_mask()`, `object_box()`, `close()`. An RL environment would call `light` → `capture` →
  a scorer.
- **`frame_row`**: turns one frame into one CSV row; this is where a different vision model would
  plug in ([vision.md](vision.md)).
- **Paths:** `POSE` (env `VRNAV_POSE`), `CONFIG` (`scene_config.json`), `OBJECT_SCALE`.
