# Data files in `training/`

## `object_classes.json`: ground truth for YOLO9000

Which YOLO9000 classes count as a correct identification of each object.

```json
{"id": 1, "object": "Mug",
 "yolo9000_accept": [{"index": 3984, "wnid": "n03797390", "name": "mug", "meaning": "drinking vessel < vessel < container"},
                     {"index": 3640, "wnid": "n03147509", "name": "cup", "meaning": "container < ..."}],
 "yolo9000_box_trained": {"node": "cup", "coco_class": "cup", "exact": true},
 "coco_class": "cup",
 "review": "\"cup\" accepted ..."}
```

| Field | Meaning |
|---|---|
| `id` | Catalogue number in the Unreal project's `Config/objectlab_objects.json` (what `spawn` takes) |
| `yolo9000_accept` | Accepted classes; each also accepts every more specific class below it. `wnid` is the WordNet ID (names repeat with different meanings); `index` is the line in Darknet's `data/9k.names`; `meaning` shows the parents, i.e. which sense of the word |
| `yolo9000_box_trained` | Nearest class at or above an accepted one that YOLO9000 learned with boxes (COCO); `null` = learned from photo labels only |
| `coco_class` | Class used for YOLO11m, or `null` |
| `review` | Notes on non-obvious choices and decisions |

`dropped` lists the 8 catalogue objects left out (no YOLO9000 synonym). How it was built and every
decision: `reports/meeting3_report_notes.md` ("Ground-truth class dataset").

**To add or change an object:** find the class in `/opt/Unrealprojects/darknet/data/9k.names`
(line number − 1 = `index`; the same line in `9k.labels` = `wnid`; `9k.tree` gives each class's
parent index), check its meaning by following parents, and add it to `yolo9000_accept`. Then
`sweep --rescore` to apply it to saved frames.

## `object_scale.json`: fixes for mis-sized models

Seven models in the Oct 5 project update are drawn at a different size from their stored bounds
(e.g. the baseball renders at 15 cm though its bounds say 7.4 cm). `Scene.load` scales these
objects by the listed factor so the drawn object matches its stored, realistic size. Measured and
verified with `check-objects` ([scene.md](scene.md)). Remove an entry once the model is fixed in
the Unreal project.

## `scene_config.json`: camera and light settings

Written by `scene.py calibrate` in the first runs; read by `sweep`, `check-objects` and
`deliverables`.

| Key | Value | Meaning |
|---|---|---|
| `camera.height_cm` | 165 | Camera height (horizontal position from `pose.json`) |
| `camera.fill` | 0.5 | Object's bounding sphere spans 50% of the frame width (zoom per object) |
| `camera.width/height` | 640 | Frame size (rendered at 2× and shrunk: `render.supersample`) |
| `light.radius_cm` | 120 | Light distance from the object |
| `render.settle_frames` | 0 | Frames to wait after moving the light (re-measured: still 0) |

Edit by hand to change a setting, or re-run `calibrate` (which overwrites it).

## `calibration.md`

The report from the first runs' `calibrate` (YOLO11m, original 10 objects): how framing and radius
were chosen. Generated; don't edit.
