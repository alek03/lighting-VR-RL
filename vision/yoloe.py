"""YOLOE-26: which of the table objects is in a picture, and how sure is it?

    from vision.yoloe import Detector

    det = Detector()                                  # loads the model and all object names once
    result = det.score(image, "Mug", true_box=box)    # image: numpy RGB array or a file path

The model is prompted with every object's name, so it has to choose between all of them.
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')    # GPU numbers as in nvidia-smi
import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402
from ultralytics import YOLOE                               # noqa: E402
from ultralytics.utils.metrics import box_iou               # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / 'config.json').read_text())


def prompt(name):
    """Catalog name -> words for the model: "AcousticGuitar" -> "acoustic guitar"."""
    fixes = CONFIG['vision']['name_fixes']
    return fixes.get(name) or re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()


def box_overlap(box, truth):
    """How YOLO's box [x0, y0, x1, y1] relates to the true object box."""
    ix = max(0.0, min(box[2], truth[2]) - max(box[0], truth[0]))
    iy = max(0.0, min(box[3], truth[3]) - max(box[1], truth[1]))
    inter = ix * iy
    area = lambda b: max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return {'iou': inter / (area(box) + area(truth) - inter),
            'object_fraction': inter / area(box)}      # low = YOLO's box is mostly not the object


class Detector:
    def __init__(self, objects=None):
        if objects is None:
            catalog = Path(CONFIG['unreal']['project']).parent / 'Config' / 'objectlab_objects.json'
            objects = json.loads(catalog.read_text())
        self.prompts = {name: prompt(name) for name in objects}
        self.names = list(self.prompts.values())
        self.model = YOLOE(str(REPO / CONFIG['vision']['model']))
        self.model.set_classes(self.names)
        self.device = 'cuda:%d' % CONFIG['gpu']

    def score(self, image, object_name, true_box=None):
        """
        verdict     'correct' (top guess is this object, on the object), 'wrong', or 'none'
                    (nothing found with at least vision.min_confidence)
        confidence  the model's confidence for this object anywhere in the picture
        top         up to 3 (name, confidence) guesses for the spot of the top guess
        box         the top guess's box; with true_box also iou and object_fraction
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)       # Ultralytics reads numpy arrays as BGR, PIL as RGB
        boxes = self.model.predict(image, conf=0.01, device=self.device, verbose=False)[0].boxes
        target = self.names.index(self.prompts[object_name])
        mine = boxes.conf[boxes.cls == target]
        result = {'verdict': 'none', 'confidence': float(mine.max()) if len(mine) else 0.0,
                  'top': [], 'box': None}
        if not len(boxes) or boxes.conf.max() < CONFIG['vision']['min_confidence']:
            return result
        order = boxes.conf.argsort(descending=True)
        best = boxes.xyxy[order[0]]
        same_spot = box_iou(best[None], boxes.xyxy)[0] > 0.7
        for i in order:
            name = self.names[int(boxes.cls[i])]
            if same_spot[i] and name not in [t[0] for t in result['top']]:
                result['top'].append((name, round(float(boxes.conf[i]), 4)))
        result['top'] = result['top'][:3]
        result['box'] = [round(float(v), 1) for v in best]
        on_object = True
        if true_box is not None:
            result.update({k: round(v, 3) for k, v in box_overlap(result['box'], true_box).items()})
            on_object = max(result['iou'], result['object_fraction']) >= CONFIG['vision']['on_object']
        right_name = result['top'][0][0] == self.prompts[object_name]
        result['verdict'] = 'correct' if right_name and on_object else 'wrong'
        return result
