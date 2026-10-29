import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl.light_env import LightSearchEnv
from rl.train_light_search import write_results


def row(verdict, confidence):
    return dict(verdict=verdict, confidence=confidence, top1_conf=confidence,
                tag='target', frame='', light_x=1, light_y=2, light_z=3)


class StopPolicy:
    def predict(self, obs, deterministic=True):
        return 4, None


class CorrectDetectionTests(unittest.TestCase):
    def test_wrong_high_confidence_does_not_beat_correct_detection(self):
        table = {('target', 0, 5): row('wrong', .99),
                 ('target', 30, 5): row('correct', .3),
                 ('target', 60, 5): row('wrong', 1)}
        env = LightSearchEnv(score_table=table)
        env.reset(options=dict(object_name='target', azimuth=0, elevation=5))
        self.assertEqual(env.confidence, 0)
        _, reward, _, _, info = env.step(1)
        self.assertAlmostEqual(reward, 1.295)
        self.assertTrue(info['correct'])
        _, reward, _, _, _ = env.step(1)
        self.assertAlmostEqual(reward, -1.305)

    def test_no_detection_is_bounded_and_penalized(self):
        env = LightSearchEnv(score_table={('target', 0, 5): row('none', 0)}, max_steps=1)
        obs, _ = env.reset(options=dict(object_name='target', azimuth=0, elevation=5))
        self.assertTrue(env.observation_space.contains(obs))
        _, reward, terminated, truncated, info = env.step(4)
        self.assertTrue(terminated and truncated)
        self.assertAlmostEqual(reward, -.1)
        self.assertFalse(info['found_correct'])

    def test_observation_has_no_object_identity_or_history(self):
        table = {('first', 0, 5): row('correct', .4),
                 ('second', 0, 5): row('correct', .4)}
        env = LightSearchEnv(score_table=table)
        first, _ = env.reset(options=dict(object_name='first', azimuth=0, elevation=5))
        second, _ = env.reset(options=dict(object_name='second', azimuth=0, elevation=5))
        import numpy as np
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (5,))
        self.assertAlmostEqual(float(first[3]), .4)
        self.assertEqual(first[4], 1)

    def test_observation_retains_confidence_when_prediction_is_wrong(self):
        env = LightSearchEnv(score_table={('target', 0, 5): row('wrong', .8)})
        obs, _ = env.reset(options=dict(object_name='target', azimuth=0, elevation=5))
        self.assertAlmostEqual(float(obs[3]), .8)
        self.assertEqual(obs[4], 0)
        self.assertEqual(env.confidence, 0)

    def test_export_never_selects_invalid_location(self):
        table = {('missing', 0, 5): row('none', 0),
                 ('wrong', 0, 5): row('wrong', .99),
                 ('valid', 0, 5): row('correct', .3)}
        env = LightSearchEnv(score_table=table)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('rl.train_light_search._plot_results'), patch('rl.train_light_search._example_contact_sheet'):
                summary = write_results(StopPolicy(), env, table, out, out, 8, 17)
            import csv
            with (out/'policy_results.csv').open() as stream:
                results = {r['object']: r for r in csv.DictReader(stream)}
            for name in ('missing', 'wrong'):
                self.assertEqual(results[name]['status'], 'no_correct_detection_in_sweep')
                self.assertEqual(results[name]['light_x_cm'], '')
                self.assertEqual(results[name]['best_policy_azimuth'], '')
            self.assertEqual(results['valid']['best_verdict'], 'correct')
            self.assertEqual(summary['objects_with_correct_detection'], 1)


if __name__ == '__main__':
    unittest.main()
