# `training/vision.py`

The vision models that judge each frame. Pin the GPU with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>` (then `cuda:0` inside the program is
that GPU).

## `Yolo9000`: the main measure

```python
from training.vision import Yolo9000
model = Yolo9000()                                  # loads Darknet + weights from DARKNET_DIR
result = model.score('frame.png', 'Mug', mask=mask) # mask: True where the object is (optional)
# result: confidence, top3 [{class, wnid, confidence, class_prob}], verdict, top_box {box, objectness, iou, ...}
```

- **Per-class probabilities.** Darknet's `detect()` keeps one label per box; the full probabilities
  for all 9,418 classes are read from the network's output buffer instead.
- **Classes are a tree** (WordNet): P(container) ≥ P(cup) ≥ P(teacup).
- **confidence** = objectness × P(accepted class), the best over all boxes.
- **top-3** = most specific classes with P ≥ `guess_thresh` (0.1), found by walking down the tree.
- **verdict** for YOLO's most confident box (`none` if its objectness < `detect_thresh`, 0.25):
  - `correct`: top guess is an accepted class or more specific, **and** the box is on the object
  - `partial`: only a more general class (e.g. "ball" for Basketball), box on the object
  - `wrong`: anything else, including any answer whose box is off the object
- **On the object** = box overlaps the mask's box with IoU ≥ 0.5, **or** at least half the box is
  object pixels (`on_object` argument). Neither (shadow, lit table, background) means wrong.
- Accepted classes per object come from `object_classes.json` ([data_files.md](data_files.md)).

Tunable in the constructor: `guess_thresh`, `detect_thresh`, `nms`, `max_boxes`; in `score`:
`on_object`.

## `Scorer`: YOLO11m (comparison)

`Scorer().score(image, 'Mug')` → highest confidence for the object's COCO class (`COCO_CLASS`,
filled from `object_classes.json`), or `None` if it has none. Weights download into `models/`.

## `box_overlap(box, truth)`

IoU, object fraction (of the box) and coverage (of the truth box) for two `[x0, y0, x1, y1]` boxes.

## Adding another model (e.g. YOLOE-26)

1. Add a class with `score(image_path, object_name, mask=None)` returning the same fields as
   `Yolo9000.score` (`confidence`, `top3`, `verdict`, `top_box`, `accepted`, `truth_box`).
2. In `training/scene.py`, create it in `sweep_sphere` and pass it to `frame_row` in place of
   `yolo9000` (or add a `--model` option).
3. Score the saved frames with `sweep --rescore --out <copy of a sweep folder>`; no rendering needed.
