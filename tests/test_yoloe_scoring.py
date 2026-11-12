import unittest

from vision.yoloe import rank_class_scores


class YoloeScoringTests(unittest.TestCase):
    def test_ground_truth_score_is_from_same_box_as_top_three(self):
        names = ['apple', 'banana', 'mango', 'lemon']
        result = rank_class_scores([0.24, 0.81, 0.63, 0.42], names, target=0)

        self.assertEqual(result['top'], [('banana', 0.81), ('mango', 0.63), ('lemon', 0.42)])
        self.assertEqual(result['confidence'], 0.81)
        self.assertEqual(result['gt_conf'], 0.24)

    def test_ground_truth_score_matches_ranked_score_when_in_top_three(self):
        names = ['apple', 'banana', 'mango']
        result = rank_class_scores([0.72, 0.51, 0.2], names, target=0)

        self.assertEqual(result['top'][0], ('apple', result['gt_conf']))
        self.assertEqual(result['confidence'], result['gt_conf'])


if __name__ == '__main__':
    unittest.main()