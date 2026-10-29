import unittest
from unittest.mock import Mock, patch

from unreal_bridge import Unreal, CONFIG


class CameraZoomTests(unittest.TestCase):
    def backend(self):
        ue = Unreal.__new__(Unreal)
        ue.config = CONFIG
        ue.centre = [1120, -250, 100]
        ue.camera = dict(CONFIG['camera'])
        ue.set_camera = Mock()
        return ue

    @patch('unreal_bridge.time.sleep')
    def test_empty_mask_after_camera_change_is_retried(self, sleep):
        ue = self.backend()
        ue.true_box = Mock(side_effect=[None, [100, 100, 400, 400]])
        ue.zoom_to_object(.5)
        self.assertEqual(ue.true_box.call_count, 2)
        self.assertEqual(ue.set_camera.call_count, 2)

    @patch('unreal_bridge.time.sleep')
    def test_persistently_empty_mask_raises_skippable_runtime_error(self, sleep):
        ue = self.backend()
        ue.true_box = Mock(return_value=None)
        with self.assertRaisesRegex(RuntimeError, 'mask stayed empty'):
            ue.zoom_to_object(.5)


if __name__ == '__main__':
    unittest.main()
