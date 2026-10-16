"""Camera invariants without an Unreal editor or vision model."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import scene


class FakeUnreal:
    def __init__(self):
        self.calls = []

    def call(self, command, **kwargs):
        self.calls.append((command, copy.deepcopy(kwargs)))
        if command == 'describe':
            return {'objects': {'Phone': {'radius_cm': 8}, 'Chair': {'radius_cm': 90}},
                    'torch_light': {}}
        if command == 'spawn':
            return {'object_bounds_center_cm': [200, 20, 80 if kwargs['object'] == 'Phone' else 130]}
        return {}


class FixedCameraTests(unittest.TestCase):
    def test_object_and_light_changes_preserve_complete_recorded_camera(self):
        camera = {'location': [10, -20, 145], 'rotation': [-13, 7, 4], 'fov': 87}
        with tempfile.TemporaryDirectory() as tmp:
            pose = Path(tmp) / 'pose.json'
            pose.write_text(json.dumps({'camera': camera}))
            scratch = Path(tmp) / 'frames'
            scratch.mkdir()
            ue = FakeUnreal()
            with patch.object(scene, 'POSE', pose), patch.object(scene.tempfile, 'mkdtemp', return_value=str(scratch)):
                capture_scene = scene.Scene(ue)
                try:
                    for name in ('Phone', 'Chair', 'Phone'):
                        capture_scene.load(name)
                        for azimuth in (0, 90, 180):
                            capture_scene.light(azimuth, 45, 120)
                        self.assertEqual(capture_scene.camera, camera)
                    setups = [args for command, args in ue.calls if command == 'capture_setup']
                    self.assertEqual(len(setups), 1)
                    self.assertEqual(setups[0]['camera'], camera)
                    self.assertEqual((setups[0]['width'], setups[0]['height']), (1280, 1280))
                    # Editing the pose during a run must not change later objects' cameras.
                    pose.write_text(json.dumps({'camera': dict(camera, fov=30)}))
                    capture_scene.load('Chair')
                    self.assertEqual(capture_scene.camera, camera)
                finally:
                    capture_scene.close()

    def test_calibration_report_uses_fixed_camera_without_legacy_fill(self):
        config = json.loads(scene.CONFIG.read_text())
        report = {'settle': {'noise_floor': 0, 'settle_frames': 0, 'diff_by_frames': {0: 0}},
                  'radius_mean_conf': {120: 0.5},
                  'radius': {'Mug': {120: {'conf': 0.5, 'brightness': 80, 'conf_spread': 0.1}}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.md'
            scene.write_report(report, config, path)
            self.assertIn('rotation and FOV unchanged', path.read_text())
            self.assertNotIn('## Framing', path.read_text())


if __name__ == '__main__':
    unittest.main()
