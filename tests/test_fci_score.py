"""The out-of-sample score (CONTEXT.md, ADR 0003 in omarzd/fci), on hand-built paths.

Every expected number here is worked out by hand from the score's definition. Most cases use one
origin of a 5-month lead, where the target path is 50, 60, 70, 80, 90 and the average guess is 40.
The guess then misses by 10 to 50 points, so each month's cost scales to 1 for the guess, and the
guess's path cost is the sum of the weights, 1 + 1/e + 1/e² + 1/e + 1.
"""
import math
import unittest

import numpy as np

from fci import StepError
from fci.score import score

TARGET = np.array([[50.0, 60, 70, 80, 90]])
GUESS = np.array([40.0])
GUESS_COST = 2 + 2 / math.e + math.e ** -2


def score_one_origin(path):
    return score(np.array([path], dtype=float), TARGET, GUESS)


class ScoreTest(unittest.TestCase):
    def test_a_perfect_path_scores_1(self):
        self.assertEqual(score_one_origin([50, 60, 70, 80, 90]), 1)

    def test_the_average_guess_scores_0(self):
        # Three origins, each with its own average guess, so the scaling averages over origins.
        target = np.array([[50.0, 58, 61], [47, 44, 52], [55, 53, 49]])
        guess = np.array([51.0, 48, 50])
        self.assertAlmostEqual(score(np.repeat(guess[:, None], 3, axis=1), target, guess), 0, places=12)

    def test_misses_of_2_points_or_less_score_as_perfect(self):
        self.assertEqual(score_one_origin([52, 58, 71.5, 80, 88]), 1)

    def test_a_miss_beyond_2_points_costs_its_excess_squared(self):
        # 3 points off at month 1: (3 - 2)² = 1, over the guess's 64 there, at weight 1.
        self.assertAlmostEqual(score_one_origin([53, 60, 70, 80, 90]), 1 - (1 / 64) / GUESS_COST, places=12)

    def test_a_path_one_month_early_or_late_scores_as_perfect(self):
        target = np.array([[50.0, 50, 60, 70, 70]])
        for path in ([50, 60, 70, 70, 70], [50, 50, 50, 60, 70]):
            with self.subTest(path=path):
                self.assertEqual(score(np.array([path], dtype=float), target, GUESS), 1)

    def test_a_path_two_months_early_or_late_does_not(self):
        # Month 1 of the early path, or month 5 of the late one, has no target month within one
        # month that it hits.
        target = np.array([[50.0, 50, 60, 70, 70]])
        for path in ([60, 70, 70, 70, 70], [50, 50, 50, 50, 60]):
            with self.subTest(path=path):
                self.assertLess(score(np.array([path], dtype=float), target, GUESS), 1)

    def test_the_end_months_weigh_e_squared_times_the_middle_month(self):
        # Each path is perfect but for one month, which is as far off as the average guess, so
        # that month costs its weight: 1 at months 1 and 5, 1/e² at month 3. The middle month's
        # neighbours are 20 points from its guess, so no warp makes it cheaper.
        for month, weight in ((1, 1), (3, math.e ** -2), (5, 1)):
            with self.subTest(month=month):
                path = [50, 60, 70, 80, 90]
                path[month - 1] = 40
                self.assertAlmostEqual(score_one_origin(path), 1 - weight / GUESS_COST, places=12)

    def test_an_average_guess_that_never_misses_a_month_is_an_error(self):
        # Within 2 points of the target at month 2 at every origin: nothing to scale that month by.
        target = np.array([[50.0, 41, 60]])
        with self.assertRaises(StepError):
            score(target, target, GUESS)


if __name__ == "__main__":
    unittest.main()
