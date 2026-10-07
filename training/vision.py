"""Vision models used as the reward: how confidently YOLO identifies the object on the table.

`Scorer` is stock YOLO11m, trained on COCO, which has no class for two of the ten original table
objects (Shoe, Lamp); `score` returns None for those.

`Yolo9000` is YOLO9000 (Darknet), whose ~9,000 classes form a WordNet tree; which classes count
as correct for each object is in training/object_classes.json.
"""
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / 'models'
DARKNET = Path(os.environ.get('DARKNET_DIR', '/opt/Unrealprojects/darknet'))
OBJECT_CLASSES = REPO / 'training' / 'object_classes.json'

# Table object -> COCO class name.
COCO_CLASS = {
    'Mug': 'cup', 'Apple': 'apple', 'Book': 'book', 'Bottle': 'bottle', 'Chair': 'chair',
    'Bowl': 'bowl', 'Phone': 'cell phone', 'Plant': 'potted plant', 'Shoe': None, 'Lamp': None,
}
# The new objects' COCO classes (where they have one) are listed in object_classes.json.
COCO_CLASS.update({o['object']: o['coco_class'] for o in json.loads(OBJECT_CLASSES.read_text())['objects']
                   if o['object'] not in COCO_CLASS})


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


class Yolo9000:
    """YOLO9000 through Darknet's Python wrapper, with per-class probabilities.

    Darknet's own `detect()` keeps one label per box. Its full class probabilities are still in
    the network's output buffer once `get_network_boxes` has run (it converts the tree's
    conditional probabilities to absolute ones in place), so they are read from there.

    Probabilities are nested: P(container) >= P(cup) >= P(teacup). A box's *guesses* are its most
    specific classes above `guess_thresh`: walk down the tree from the root, following every
    child at or above the threshold, and keep the nodes where no child is.

    Uses CUDA device 0 of the process; pin it with CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES=<gpu>.
    """
    COORDS, ANCHORS = 4, 3

    def __init__(self, guess_thresh=0.1, detect_thresh=0.25, nms=0.45, max_boxes=20):
        sys.path.insert(0, str(DARKNET / 'python'))
        cwd = os.getcwd()
        os.chdir(DARKNET)                       # the cfg names data/9k.tree relative to here
        try:
            import darknet
            self.dn = darknet
            self.net = darknet.load_net(b'cfg/yolo9000.cfg', b'yolo9000-weights/yolo9000.weights', 0)
        finally:
            os.chdir(cwd)
        self.names = [l.rstrip('\n') for l in open(DARKNET / 'data' / '9k.names')]
        self.wnids = [l.strip() for l in open(DARKNET / 'data' / '9k.labels')]
        self.parent = [int(l.split()[1]) for l in open(DARKNET / 'data' / '9k.tree')]
        self.children = [[] for _ in self.parent]
        for c, p in enumerate(self.parent):
            if p >= 0:
                self.children[p].append(c)
        self.roots = [c for c, p in enumerate(self.parent) if p < 0]
        self.grid = darknet.lib.network_width(self.net) // 32
        self.guess_thresh, self.detect_thresh, self.nms, self.max_boxes = guess_thresh, detect_thresh, nms, max_boxes
        self.objects = {o['object']: o for o in json.loads(OBJECT_CLASSES.read_text())['objects']}

    def ancestors(self, c):
        out = []
        while c >= 0:
            out.append(c)
            c = self.parent[c]
        return out

    def guesses(self, probs, k=3):
        found, stack = [], [c for c in self.roots if probs[c] >= self.guess_thresh]
        while stack:
            c = stack.pop()
            kids = [d for d in self.children[c] if probs[d] >= self.guess_thresh]
            if kids:
                stack.extend(kids)
            else:
                found.append(c)
        return sorted(found, key=lambda c: -probs[c])[:k]

    def boxes(self, image_path):
        """Candidate boxes after non-max suppression, most confident first. Each: corner box in
        pixels, objectness, and absolute P(class | object) for all classes."""
        dn, wh = self.dn, self.grid * self.grid
        im = dn.load_image(str(image_path).encode(), 0, 0)
        out = dn.predict_image(self.net, im)
        num = ctypes.c_int(0)
        dets = dn.get_network_boxes(self.net, im.w, im.h, 0.005, 0.5, None, 0, ctypes.pointer(num))
        try:
            n_classes = len(self.names)
            buf = np.ctypeslib.as_array(out, shape=(self.ANCHORS * wh * (self.COORDS + 1 + n_classes),))
            buf = buf.reshape(self.ANCHORS, self.COORDS + 1 + n_classes, wh)
            objectness = buf[:, self.COORDS, :].reshape(-1)          # index = anchor * wh + cell
            kept = []
            for k in np.argsort(-objectness):
                if objectness[k] < 0.005 or len(kept) == self.max_boxes:
                    break
                b = dets[int(k)].bbox
                box = [max(0.0, b.x - b.w / 2), max(0.0, b.y - b.h / 2), min(im.w, b.x + b.w / 2), min(im.h, b.y + b.h / 2)]
                if all(box_overlap(box, other['box'])['iou'] <= self.nms for other in kept):
                    n, cell = divmod(int(k), wh)
                    kept.append({'box': box, 'objectness': float(objectness[k]),
                                 'probs': buf[n, self.COORDS + 1:, cell].copy()})
            return kept
        finally:
            dn.free_detections(dets, num.value)
            dn.free_image(im)

    def score(self, image_path, object_name, mask=None, on_object=0.5):
        """Everything recorded for one frame. `mask` (True where the object is, if known) enables
        the box check: a top answer only counts when its box is on the object, meaning its overlap
        with the object's box (IoU) or the share of the box that is object pixels (object
        fraction) is at least `on_object`. A box that is neither, e.g. around a shadow or the lit
        table, makes the answer wrong. Object fraction alone would fail thin or open objects
        (scissors), IoU alone fails boxes around a clear part of the object (a plant's pot)."""
        accept = {a['index'] for a in self.objects[object_name]['yolo9000_accept']}
        accepted_or_below = lambda c: any(a in accept for a in self.ancestors(c))
        above_accepted = {a for c in accept for a in self.ancestors(c)[1:]}
        boxes = self.boxes(image_path)

        def own_conf(b):
            return b['objectness'] * max(float(b['probs'][c]) for c in accept)

        result = {'object': object_name, 'accepted': sorted(self.names[c] for c in accept),
                  'confidence': max((own_conf(b) for b in boxes), default=0.0)}
        top = boxes[0] if boxes else None
        if top is None or top['objectness'] < self.detect_thresh:
            result.update(verdict='none', top3=[], top_box=None)
            return result
        guesses = self.guesses(top['probs'])
        result['top3'] = [{'class': self.names[c], 'wnid': self.wnids[c],
                           'confidence': round(top['objectness'] * float(top['probs'][c]), 4),
                           'class_prob': round(float(top['probs'][c]), 4)} for c in guesses]
        result['top_box'] = {'box': [round(v, 1) for v in top['box']], 'objectness': round(top['objectness'], 4)}
        is_on = True
        if mask is not None and mask.any():
            ys, xs = np.nonzero(mask)
            truth = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            x0, y0, x1, y1 = (int(round(v)) for v in top['box'])
            inside = float(mask[y0:y1, x0:x1].sum())
            result['truth_box'] = truth
            result['top_box'].update(iou=round(box_overlap(top['box'], truth)['iou'], 3),
                                     object_fraction=round(inside / max(1, (x1 - x0) * (y1 - y0)), 3),
                                     coverage=round(inside / mask.sum(), 3),
                                     centre_in_box=int(x0 <= (truth[0] + truth[2]) / 2 <= x1 and y0 <= (truth[1] + truth[3]) / 2 <= y1))
            is_on = max(result['top_box']['iou'], result['top_box']['object_fraction']) >= on_object
        if not is_on:
            verdict = 'wrong'                    # the answer is about something else in the frame
        elif not guesses or guesses[0] in above_accepted:
            verdict = 'partial'                  # on the object, but only a more general class
        elif accepted_or_below(guesses[0]):
            verdict = 'correct'
        else:
            verdict = 'wrong'
        result['verdict'] = verdict
        return result
