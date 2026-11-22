"""E0: exact finite-horizon Bayesian decision planning (stdlib only).

Hidden state theta in {0,1}; the agent may commit to a prediction or pay
for a noisy observation. Questions are single-use and have known likelihoods.
All expected values are measured under the predeclared utility function.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


@dataclass(frozen=True)
class Signal:
    name: str
    yes_if_1: float
    yes_if_0: float
    cost: float

    def __post_init__(self):
        if not 0 <= self.yes_if_1 <= 1 or not 0 <= self.yes_if_0 <= 1:
            raise ValueError("likelihoods must be in [0,1]")
        if self.cost < 0:
            raise ValueError("observation costs cannot be negative")


@dataclass(frozen=True)
class Task:
    prior: float
    signals: Tuple[Signal, ...]
    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    gamma: float = 1.0
    max_queries: int = 3

    def __post_init__(self):
        if not 0 <= self.prior <= 1:
            raise ValueError("prior must be in [0,1]")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0,1]")
        if self.max_queries < 0:
            raise ValueError("max_queries must be nonnegative")
        if len({x.name for x in self.signals}) != len(self.signals):
            raise ValueError("signal names must be unique")


def response_probability(p, signal, answer):
    """P(answer | prior p), marginalizing the latent state."""
    a = signal.yes_if_1 if answer else 1 - signal.yes_if_1
    b = signal.yes_if_0 if answer else 1 - signal.yes_if_0
    return p * a + (1 - p) * b


def update(p, signal, answer):
    """Exact Bayes update; never conditions on an impossible observation."""
    likelihood = signal.yes_if_1 if answer else 1 - signal.yes_if_1
    mass = response_probability(p, signal, answer)
    if mass <= 0:
        raise ValueError("zero-probability observation")
    return p * likelihood / mass


def commit_value(task, p, choice):
    correct_prob = p if choice == 1 else 1 - p
    return correct_prob * task.reward_correct + (1 - correct_prob) * task.reward_wrong


def commit_action(task, p):
    values = [commit_value(task, p, choice) for choice in (0, 1)]
    return ("commit", 1 if values[1] > values[0] else 0)


def terminal_value(task, p):
    return max(commit_value(task, p, 0), commit_value(task, p, 1))


def _query_value(task, p, idx, remaining, depth, continuation):
    signal = task.signals[idx]
    future = tuple(k for k in remaining if k != idx)
    total = -signal.cost
    for answer in (False, True):
        prob = response_probability(p, signal, answer)
        if prob > 0:
            posterior = update(p, signal, answer)
            total += task.gamma * prob * continuation(posterior, future, depth - 1)
    return total


def choose(task, p, remaining, depth, mode="exact"):
    """Return (action, value estimated by this planning method).

    greedy = immediately commit; one_step = query assuming immediate
    commitment after this query; exact = recurse over the remaining horizon.
    """
    if mode not in {"greedy", "one_step", "exact"}:
        raise ValueError(mode)
    if mode == "exact":
        return _exact_choose(task, p, remaining, depth)
    best = commit_action(task, p)
    value = terminal_value(task, p)
    if mode == "greedy" or depth <= 0:
        return best, value
    for idx in remaining:
        qval = _query_value(task, p, idx, remaining, depth,
                            lambda pp, rest, d: terminal_value(task, pp))
        if qval > value + 1e-12:
            best, value = ("query", idx), qval
    return best, value


@lru_cache(maxsize=None)
def _exact_choose(task, p, remaining, depth):
    best = commit_action(task, p)
    value = terminal_value(task, p)
    if depth <= 0:
        return best, value
    for idx in remaining:
        qval = _query_value(task, p, idx, remaining, depth,
                            lambda pp, rest, d: _exact_choose(task, pp, rest, d)[1])
        if qval > value + 1e-12:
            best, value = ("query", idx), qval
    return best, value


def evaluate(task, mode="exact", trace=False):
    """Exact expected return of a deployed re-planning policy (no sampling).

    Records complete action/observation branches if trace=True. Each leaf
    includes its probability and discounted realized reward conditional on
    the history; no hidden state is provided to the policy.
    """
    audit = []

    def recurse(p, remaining, depth, mass, path):
        action, predicted = choose(task, p, remaining, depth, mode)
        if trace:
            audit.append({"history": list(path), "belief_1": round(p, 12),
                          "action": action, "prediction": round(predicted, 12),
                          "path_probability": round(mass, 12),
                          "forecast_frozen_before_action": True})
        if action[0] == "commit":
            return commit_value(task, p, action[1])
        idx = action[1]
        sig = task.signals[idx]
        remaining = tuple(k for k in remaining if k != idx)
        result = -sig.cost
        for y in (False, True):
            prob = response_probability(p, sig, y)
            if prob > 0:
                post = update(p, sig, y)
                result += task.gamma * prob * recurse(
                    post, remaining, depth - 1, mass * prob,
                    path + ((sig.name, int(y)),))
        return result

    value = recurse(task.prior, tuple(range(len(task.signals))),
                    task.max_queries, 1.0, ())
    return (value, audit) if trace else value
