"""YOLOE tag-conditioned detection and object-mask scoring."""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / 'models'
OBJECT_CLASSES = REPO / 'training' / 'object_classes.json'
CATALOG = json.loads(OBJECT_CLASSES.read_text())['objects']
COCO_CLASS = {o['object']: o['coco_class'] for o in CATALOG}
YOLOE_LABELS = {o['object']: o['tag'] for o in CATALOG}


class Scorer:
    def __init__(self, weights='yolo11m.pt', device='cuda:0'):
        from ultralytics import YOLO
        MODELS.mkdir(exist_ok=True)
        # A missing file under models/ is downloaded there, keeping weights out of the repo root.
        self.model = YOLO(str(MODELS / weights))
        self.device = device
        self.index = {name: i for i, name in self.model.names.items()}

    def score(self, image, object_name):
        """Highest confidence for the object's class anywhere in the image (0 if not found)."""
        cls = COCO_CLASS[object_name]
        if cls is None:
            return None
        result = self.model.predict(image, device=self.device, conf=0.001, verbose=False,
                                    classes=[self.index[cls]])[0]
        return float(result.boxes.conf.max()) if len(result.boxes) else 0.0


def box_overlap(box, truth):
    """How a predicted box [x0, y0, x1, y1] relates to the true object box."""
    ix = max(0.0, min(box[2], truth[2]) - max(box[0], truth[0]))
    iy = max(0.0, min(box[3], truth[3]) - max(box[1], truth[1]))
    inter = ix * iy
    area = lambda b: max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return {'iou': inter / (area(box) + area(truth) - inter),
            'object_fraction': inter / area(box),      # low = box is mostly shadow or background
            'coverage': inter / area(truth)}           # low = box catches only part of the object


class YoloE26:
    """YOLOE with one fixed catalogue of text-label prompts."""

    def __init__(self, weights='yoloe-26x-seg.pt', device='cuda:0', detect_thresh=0.25, conf=0.001):
        from ultralytics import YOLOE
        self.labels = YOLOE_LABELS
        self.names = list(dict.fromkeys(self.labels.values()))
        self.device, self.detect_thresh, self.conf = device, detect_thresh, conf
        MODELS.mkdir(exist_ok=True)
        self.model = YOLOE(str(MODELS / weights)).to(device)
        self.model.set_classes(self.names)

    def boxes(self, image):
        if isinstance(image, np.ndarray):
            image = np.ascontiguousarray(image[..., ::-1])
        result = self.model.predict(str(image) if isinstance(image, Path) else image,
                                    device=self.device, conf=self.conf, max_det=300,
                                    verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.cpu().numpy()
        return sorted([{'box': [float(v) for v in box], 'class': self.names[int(cls)],
                        'confidence': float(conf)}
                       for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf)],
                      key=lambda item: -item['confidence'])

    def score(self, image, object_name, mask=None, on_object=0.5):
        label, boxes = self.labels[object_name], self.boxes(image)
        own = [b for b in boxes if b['class'] == label]
        result = {'object': object_name, 'tag': label,
                  'confidence': max((b['confidence'] for b in own), default=0.0),
                  'verdict': 'none', 'top3': [], 'top_box': None}
        truth = None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.ndim != 2:
                raise ValueError('mask must be a 2D object silhouette')
            ys, xs = np.nonzero(mask)
            if len(xs):
                truth = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                result['truth_box'] = truth
        if not boxes or boxes[0]['confidence'] < self.detect_thresh:
            return result
        seen = set()
        for box in boxes:
            if box['confidence'] >= self.detect_thresh and box['class'] not in seen:
                result['top3'].append({'class': box['class'], 'confidence': round(box['confidence'], 4)})
                seen.add(box['class'])
                if len(seen) == 3:
                    break
        top = boxes[0]
        result['top_box'] = {'box': [round(v, 1) for v in top['box']]}
        is_on = mask is None
        if truth is not None:
            h, w = mask.shape
            x0, y0, x1, y1 = (int(round(v)) for v in top['box'])
            inside = float(mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].sum())
            metrics = {'iou': box_overlap(top['box'], truth)['iou'],
                       'object_fraction': inside / max(1, (x1 - x0) * (y1 - y0)),
                       'coverage': inside / mask.sum(),
                       'centre_in_box': int(x0 <= (truth[0] + truth[2]) / 2 <= x1 and
                                            y0 <= (truth[1] + truth[3]) / 2 <= y1)}
            result['top_box'].update({k: round(float(v), 3) for k, v in metrics.items()})
            is_on = max(metrics['iou'], metrics['object_fraction']) >= on_object
        result['verdict'] = 'correct' if is_on and top['class'] == label else 'wrong'
        return result
