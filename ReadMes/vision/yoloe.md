# vision/yoloe.py

Scores a picture with YOLOE-26: which of the table objects does it see, and how sure is it?

```python
from vision.yoloe import Detector

det = Detector()                                  # loads the model and all object names once (~13 s)
result = det.score(image, "Mug", true_box=box)    # image: numpy RGB array or a file path
```

## Detector

| Call | What it does |
|---|---|
| `Detector(objects=None)` | Loads the model from `vision.model` and gives it every object name as a prompt. By default the names are the project's object list; pass a list to use others. |
| `det.score(image, object_name, true_box=None)` | Scores one picture for the object that is on the table. |

`score` returns:

| Key | Meaning |
|---|---|
| `verdict` | `correct` (top guess is this object, and its box is on the object), `wrong`, or `none` (nothing found with at least `vision.min_confidence`) |
| `confidence` | The model's confidence for this object anywhere in the picture (0 if never) |
| `top` | Up to 3 `(name, confidence)` guesses for the spot of the top guess |
| `box` | The top guess's box `[x0, y0, x1, y1]` |
| `iou`, `object_fraction` | Only with `true_box`: how YOLOE's box overlaps the true box |

"Box on the object" means IoU or object fraction is at least `vision.on_object` (0.5). Without
`true_box`, only the name is checked.

## How object names become prompts

`prompt(name)` turns a project name into words: it puts a space before each capital letter and
lowercases (`"AcousticGuitar"` → `"acoustic guitar"`). A name in `vision.name_fixes` uses the
text given there instead. All the phrases go to `set_classes`, and YOLOE can then only answer with
one of them.

## Plain functions

| Function | What it does |
|---|---|
| `prompt(name)` | Project name → prompt text (above). |
| `box_overlap(box, truth)` | `iou` (overlap ÷ both boxes together) and `object_fraction` (overlap ÷ YOLOE's box; low = the box is mostly not the object). |

## Settings used (config.json)

`vision` (`model`, `min_confidence`, `on_object`, `name_fixes`), `gpu`, `unreal.project` (for the
object list).

## Good to know

- Model files live in `models/`. The text model (`mobileclip2_b.ts`) is found through Ultralytics'
  `weights_dir` setting, which points at `models/`.
- After loading, a picture takes about 0.05 s.
