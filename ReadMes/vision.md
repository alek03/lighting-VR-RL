# YOLOE-26n (default scorer)

`YoloE26n` in `training/vision.py` loads `models/yoloe-26n-seg.pt` through Ultralytics.
Install/update dependencies with `.venv/bin/pip install -U -r requirements.txt`.
YOLOE-26 requires Ultralytics >=8.4.0. The first run downloads weights and the text
encoder and may install Ultralytics CLIP; network access is needed for this setup.
See the [official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/).

```python
from training.vision import YoloE26n
scorer = YoloE26n(device='cuda:0')  # 'cpu' is also supported
result = scorer.score('frame.png', 'Mug', mask=object_mask)
```

`yoloe_labels()` turns the complete object catalogue into readable text labels, with
explicit overrides for cell phone, potted plant and computer keyboard. All labels are
set once, regardless of `--objects` or the ground-truth object being evaluated. The
silhouette is used for scoring only, never as a prompt. Paths, PIL images and RGB
NumPy arrays are supported. Captures are not cropped to the object.

- `confidence`: maximum returned detection confidence for the expected label, or zero.
- `top3`: up to three distinct detected labels ranked by each label's best box confidence;
  these are not three alternative class probabilities for one box.
- `top_box`: highest-confidence detection in the full image. The existing mask overlap
  rule determines whether it lies on the object. An empty supplied mask cannot yield correct.
- Verdicts: `correct`, `wrong`, or `none` (below detection threshold). No WordNet `partial`.
- `top1_wnid` and `objectness` are blank: YOLOE does not expose these Darknet quantities.

The segmentation checkpoint supplies the detections; Unreal's independent mask remains
the ground truth. YOLO11m remains the auxiliary COCO scorer and calibration model.
YOLO9000 is retained via `--model yolo9000`; its legacy API is documented below.

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
