"""The out-of-sample score of a lead's predicted paths (CONTEXT.md, ADR 0003 in omarzd/fci).

At each graded origin the fit predicts the target months 1 to L after it. Month h weighs
e^-min(h-1, L-h), so the first and last months weigh most. A miss costs (|miss| - 2)² beyond 2
points and nothing within them. Each month's cost is divided by the average guess's mean cost at
that month over the origins, the average guess at an origin being the target's mean through it.
A path's cost is its dynamic-time-warping cost against the target path with a one-month window,
each matched pair costing the target month's weight times its scaled miss. The score is 1 minus
the fit's total path cost over the average guess's: 1 is perfect, 0 is as good as the guess.
"""
import numpy as np

from fci import StepError

SLACK = 2  # points a miss may be off at no cost
WINDOW = 1  # months a predicted month may match the target early or late


def miss_cost(miss):
    return np.maximum(np.abs(miss) - SLACK, 0) ** 2


def path_cost(pair_cost):
    """The cheapest warping path from the first months to the last, within the window. pair_cost[i, j]
    is the cost of matching predicted month i to target month j."""
    n = len(pair_cost)
    total = np.full((n + 1, n + 1), np.inf)
    total[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - WINDOW), min(n, i + WINDOW) + 1):
            total[i, j] = pair_cost[i - 1, j - 1] + min(total[i - 1, j], total[i, j - 1], total[i - 1, j - 1])
    return total[n, n]


def score(predicted, actual, guess):
    """predicted and actual: one row per origin, the target months 1 to L after it. guess: the
    average guess at each origin."""
    lead = actual.shape[1]
    h = np.arange(1, lead + 1)
    weight = np.exp(-np.minimum(h - 1, lead - h))
    scale = miss_cost(guess[:, None] - actual).mean(axis=0)
    if (scale == 0).any():
        raise StepError(f"the average guess is within {SLACK} points of the target at month "
                        f"{h[scale == 0][0]} at every graded origin, so the score can't scale that month")

    def cost(path, target):  # rows: predicted months, columns: target months
        return path_cost(weight * miss_cost(path[:, None] - target) / scale)

    fit_cost = sum(cost(p, a) for p, a in zip(predicted, actual))
    guess_cost = sum(cost(np.full(lead, g), a) for g, a in zip(guess, actual))
    return float(1 - fit_cost / guess_cost)
