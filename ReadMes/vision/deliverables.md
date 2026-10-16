# vision/deliverables.py

Turns a finished, scored sweep into figures for presenting. Doesn't use Unreal or YOLO: it reads
`sweep.csv`, `settings.json` and the pictures in `frames/`.

```
cd /home/axm6766/lighting-rl
.venv/bin/python -m vision.deliverables outputs/sweep_2026-10-17     # ~4.5 min for all objects
```

Figures go to `<sweep>/deliverables/`. Options: `--out folder` (save elsewhere),
`--best-worst 4` (objects in `3_best_vs_worst.png`, default 6).

## Figures

| File | What it shows |
|---|---|
| `objects/<Object>/1_lighting_map_<Object>.png` | Top view with the camera, the object and the 60 light positions. Left: coloured by confidence. Right: correct / wrong / nothing found. |
| `objects/<Object>/2_image_grid_<Object>.png` | All 60 pictures (rows = light height, columns = direction), with YOLOE's box, the true box (dashed magenta) and the top guess. |
| `3_best_vs_worst.png` | The objects where light matters most: best-lit vs worst-lit picture. |
| `4_recognition_map.png` | Every object: how often it is correct (left to right) vs how confident (bottom to top). Splits into never / lighting-dependent / always recognised. |
| `6_average_lighting_map.png` | Like the lighting map, averaged over all objects: which light positions work best overall. |
| `0_overview.png` | Thumbnails of the above with their file names. |

## Where the drawn positions come from

- The camera, light distance and table size: the sweep's `settings.json`, so the figures show the
  setup the sweep was actually made with, even after `config.json` changes.
- The object's position: the average of the recorded light positions (they form a dome around it).
- The table outline is only drawn if `table.size_cm` was set when the sweep was made.
