# `training/deliverables.py`

Builds the figures from a sweep. No Unreal or GPU needed.

```bash
.venv/bin/python -m training.deliverables captures/sweep_mine --out deliverables_mine
```

| Option | Default | Meaning |
|---|---|---|
| sweep folder | (required) | A `sweep` output folder (`sweep.csv`, `frames/`) |
| `--out` | `deliverables/` | Where to write the figures |
| `--best-worst` | 6 | Objects shown in the best-vs-worst figure |

**Output** (about 170 MB for 107 objects, 5 minutes):

```
0_overview.png                         thumbnail of every figure next to its file name
3_best_vs_worst.png                    objects with the largest confidence range: best vs worst frame
4_recognition_map.png                  every object by accuracy (x) and confidence (y)
6_average_lighting_map.png             top view: mean confidence and accuracy per light position
objects/<Object>/1_lighting_map_<Object>.png   top view: confidence and result per light position
objects/<Object>/2_image_grid_<Object>.png     all 60 frames with YOLO's box and the true box
```

Numbers match the deliverable list in `reports/meeting3_report_notes.md` (5 and 7 were dropped).

## Figure conventions

- **Top-view room maps** (1, 6): camera at the left with its view lines, table, the 120 cm light
  circle, one dot per light position at its true top-down location (+Y down, as in the Unreal
  editor). Higher lights sit closer to the object in this view.
- **Verdict colours/markers:** green circle = correct, amber triangle = partial, red ✗ = wrong,
  hollow grey = no detection.
- **Boxes in frames:** dashed magenta = true object box (from the mask); solid green/amber/red =
  YOLO's box.
- **Recognition map colours:** blue = YOLO9000 learned the class with boxes; orange = photo labels
  only (`yolo9000_box_trained` in `object_classes.json`).
- **Average map colour scale** runs from 0 to just above the best position (shown on the colour
  bar), because averages over many objects span a narrow range.

## Editing

- Room geometry for the maps: `TABLE`, `OBJ`, `RADIUS` at the top (measured from
  `captures/room_map.png`; 1 px = 1 cm). The camera position is read from `VRNAV_POSE`.
- Each figure is one function (`lighting_map`, `image_grid`, `best_worst`, `recognition_map`,
  `average_map`, `overview`), called from `main()`.
