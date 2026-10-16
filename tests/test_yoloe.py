import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from training.vision import YoloE26n, make_scorer, yoloe_labels
from training.scene import frame_row


class YoloETests(unittest.TestCase):
    def setUp(self):
        self.scorer = YoloE26n.__new__(YoloE26n)
        self.scorer.labels = {'Mug': 'mug', 'Chair': 'chair'}
        self.scorer.detect_thresh = 0.25
        self.mask = np.zeros((20, 20), dtype=bool)
        self.mask[5:15, 5:15] = True

    def score(self, boxes, mask=None):
        self.scorer.boxes = lambda image: boxes
        return self.scorer.score('frame.png', 'Mug', mask=mask)

    def test_correct_wrong_empty_and_missing_mask(self):
        box = {'box': [5, 5, 15, 15], 'class': 'mug', 'confidence': 0.8}
        self.assertEqual(self.score([box], self.mask)['verdict'], 'correct')
        self.assertEqual(self.score([dict(box, **{'class': 'chair'})], self.mask)['verdict'], 'wrong')
        self.assertEqual(self.score([dict(box, box=[0, 0, 3, 3])], self.mask)['verdict'], 'wrong')
        self.assertEqual(self.score([box], np.zeros_like(self.mask))['verdict'], 'wrong')
        self.assertEqual(self.score([], self.mask)['verdict'], 'none')
        self.assertEqual(self.score([dict(box, confidence=0.1)], self.mask)['confidence'], 0.1)

    def test_row_has_no_invented_wordnet_or_objectness(self):
        self.score([{'box': [5, 5, 15, 15], 'class': 'mug', 'confidence': 0.8}])
        row = frame_row(self.scorer, 'frame.png', 'Mug', self.mask, {})
        self.assertEqual(row['top1'], 'mug')
        self.assertEqual(row['top1_wnid'], '')
        self.assertNotIn('objectness', row)
        self.assertEqual(row['iou'], 1)

    def test_initialization_uses_whole_catalogue_once_and_device(self):
        model = MagicMock()
        factory = MagicMock(return_value=model)
        with tempfile.TemporaryDirectory() as tmp, patch('training.vision.MODELS', Path(tmp)), \
                patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLOE=factory)}):
            scorer = make_scorer(device='cpu')
        self.assertIsInstance(scorer, YoloE26n)
        self.assertTrue(factory.call_args.args[0].endswith('yoloe-26n-seg.pt'))
        model.to.assert_called_once_with('cpu')
        model.set_classes.assert_called_once_with(list(dict.fromkeys(yoloe_labels().values())))
        self.assertGreater(len(scorer.names), 100)

    def test_prediction_coordinates_confidence_and_rgb(self):
        self.scorer.names = ['mug', 'chair']
        self.scorer.device, self.scorer.conf, self.scorer.max_boxes = 'cpu', 0.001, 300
        boxes = MagicMock()
        boxes.cpu.return_value.numpy.return_value = SimpleNamespace(
            xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), cls=np.array([0, 1]), conf=np.array([0.4, 0.9]))
        self.scorer.model = MagicMock()
        self.scorer.model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        result = self.scorer.boxes(np.array([[[255, 0, 0]]], dtype=np.uint8))
        self.assertEqual(result[0]['class'], 'chair')
        self.assertEqual(result[0]['box'], [5, 6, 7, 8])
        np.testing.assert_array_equal(self.scorer.model.predict.call_args.args[0], [[[0, 0, 255]]])
        self.assertNotIn('classes', self.scorer.model.predict.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
