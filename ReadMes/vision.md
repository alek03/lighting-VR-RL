# YOLOE vision scoring

`training/vision.py` uses YOLOE-26n with fixed text prompts from `training/object_classes.json`.
The 115 prompt tags use the object labels, with readable spacing and a few explicit names such as
`cell phone`, `potted plant`, `computer keyboard`, and `reflex camera`.

```python
from training.vision import YoloE26n
scorer = YoloE26n(device='cuda:0')
result = scorer.score('frame.png', 'Mug', mask=object_mask)
```

`confidence` is YOLOE's highest detection confidence for the object's tag. `top3` lists up to
three different detected tags. `verdict` is `correct` when the top detection has the expected tag
and overlaps the object mask; it is `wrong` for another tag or an off-object detection, and `none`
when no detection clears the threshold. Mask overlap uses IoU or the fraction of the predicted box
covered by object pixels, with a default threshold of 0.5.

The mask is evaluation ground truth only; it is never sent as a detector prompt. RGB NumPy images,
PIL images, and image paths are supported. The checkpoint is `models/yoloe-26n-seg.pt`.
