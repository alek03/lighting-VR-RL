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
from ultralytics.models.yolo.segment import SegmentationPredictor  # noqa: E402
from unreal_bridge import object_catalog                 # noqa: E402

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


def rank_class_scores(class_scores, names, target):
    ranked = sorted(enumerate(class_scores), key=lambda item: item[1], reverse=True)
    return {'confidence': round(float(ranked[0][1]), 4),
            'gt_conf': round(float(class_scores[target]), 4),
            'top': [(names[index], round(float(conf), 4)) for index, conf in ranked[:3]]}


class ClassScorePredictor(SegmentationPredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        raw = preds[1] if isinstance(preds, (tuple, list)) and len(preds) > 1 else None
        raw = raw.get('one2one', raw) if isinstance(raw, dict) else None
        scores = raw.get('scores') if raw else None
        class_scores = None
        if scores is not None:
            scores = scores.sigmoid()
            best_box = scores.max(dim=1).values.argmax(dim=1)
            class_scores = [scores[i, :, best_box[i]].detach().cpu().tolist() for i in range(len(scores))]

        results = super().postprocess(preds, img, orig_imgs, **kwargs)
        if class_scores is not None:
            for result, scores_for_box in zip(results, class_scores):
                result.class_scores = scores_for_box
        return results


class Detector:
    def __init__(self, objects=None, collection='common'):
        if objects is None:
            objects = object_catalog(collection)
        self.prompts = {name: prompt(name) for name in objects}
        self.names = list(self.prompts.values())
        self.model = YOLOE(str(REPO / CONFIG['vision']['model']))
        self.model.set_classes(self.names)
        self.device = 'cuda:%d' % CONFIG['gpu']
        self.model.predictor = ClassScorePredictor(
            overrides={'task': self.model.task, 'mode': 'predict', 'device': self.device, 'verbose': False},
            _callbacks=self.model.callbacks)
        self.model.predictor.setup_model(model=self.model.model, verbose=False)

    def score(self, image, object_name, true_box=None):
        """
        verdict     'correct' (top guess is this object, on the object), 'wrong', or 'none'
                    (no box returned by the model)
        gt_conf     ground-truth class score on the highest-scoring box
        top         up to 3 (name, confidence) class scores on that same box
        box         the highest-scoring box; with true_box also iou and object_fraction
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)       # Ultralytics reads numpy arrays as BGR, PIL as RGB
        prediction = self.model.predict(image, conf=0.0, device=self.device, verbose=False)[0]
        target = self.names.index(self.prompts[object_name])
        ranked = rank_class_scores(prediction.class_scores, self.names, target)
        result = {'verdict': 'none', **ranked, 'box': None}
        if not len(prediction.boxes):
            return result
        result['box'] = [round(float(v), 1) for v in prediction.boxes.xyxy[0]]
        on_object = True
        if true_box is not None:
            result.update({k: round(v, 3) for k, v in box_overlap(result['box'], true_box).items()})
            on_object = max(result['iou'], result['object_fraction']) >= CONFIG['vision']['on_object']
        right_name = result['top'][0][0] == self.prompts[object_name]
        result['verdict'] = 'correct' if right_name and on_object else 'wrong'
        return result
