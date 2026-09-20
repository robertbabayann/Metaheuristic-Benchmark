import glob
import json
import os

import numpy as np


class BudgetExceeded(Exception):
    pass


class QueryBudget:
    def __init__(self, benchmark, budget):
        self.benchmark = benchmark
        self.budget = int(budget)
        self.count = 0
        self.best = float("inf")
        self.best_history = []

    def __call__(self, vector):
        if self.count >= self.budget:
            raise BudgetExceeded
        value = self.benchmark.evaluate(vector)
        self.count += 1
        if value < self.best:
            self.best = value
        self.best_history.append(self.best)
        return value


def random_search_best(benchmark, budget, seed):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(budget):
        vector = rng.uniform(0.0, 1.0, benchmark.encoding_dim)
        score = benchmark.evaluate(vector)
        if best is None or score < best:
            best = score
    return best


def random_baseline_median(benchmark, budget, n_repeats, seed):
    rng = np.random.default_rng(seed)
    repeat_seeds = rng.integers(0, 2**31 - 1, size=n_repeats)
    bests = [random_search_best(benchmark, budget, int(s)) for s in repeat_seeds]
    return float(np.median(bests))


def compute_targets(benchmark, random_median, target_levels):
    optimum = benchmark.global_optimum()
    gap = random_median - optimum
    targets = {}
    for i, level in enumerate(target_levels, start=1):
        targets[f"target_{i}"] = random_median - level * gap
    targets["optimum"] = optimum
    targets["random_median"] = random_median
    return targets


def queries_to_targets(best_history, targets):
    result = {}
    for key, threshold in targets.items():
        if not key.startswith("target_"):
            continue
        hit = next((i + 1 for i, value in enumerate(best_history) if value <= threshold), None)
        result[key] = hit if hit is not None else float("inf")
    return result


def performance_profile(cost_table, taus):
    methods = list(cost_table.keys())
    problems = sorted({problem for problems in cost_table.values() for problem in problems})

    ratios = {method: [] for method in methods}
    for problem in problems:
        values = {method: cost_table[method][problem] for method in methods if problem in cost_table[method]}
        if not values:
            continue
        finite_values = [v for v in values.values() if np.isfinite(v)]
        if not finite_values:
            continue
        best = min(finite_values)
        for method, v in values.items():
            ratios[method].append(v / best if np.isfinite(v) else np.inf)

    profiles = {}
    for method in methods:
        r = np.array(ratios[method])
        if r.size == 0:
            profiles[method] = np.zeros_like(taus, dtype=float)
            continue
        profiles[method] = np.array([np.mean(r <= tau) for tau in taus])
    return profiles


def load_cost_table(results_dir, metric_keys):
    cost_table = {}
    pattern = os.path.join(results_dir, "*", "*", "*.json")
    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)

        method_key = f"{data['algorithm']}_{data['mode']}"
        track = path.split(os.sep)[-3]
        dataset = path.split(os.sep)[-2]

        for metric_key in metric_keys:
            for record in data["records"]:
                problem_key = f"{track}/{dataset}/seed{record['seed']}:{metric_key}"
                cost_table.setdefault(method_key, {})[problem_key] = record[metric_key]

    return cost_table
