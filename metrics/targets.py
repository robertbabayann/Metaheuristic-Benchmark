import numpy as np

from experiment.progress import ProgressTracker


def random_search_best(benchmark, budget, seed, desc=None):
    rng = np.random.default_rng(seed)
    best = None
    for i in range(budget):
        if desc:
            ProgressTracker.desc(f"{desc} | eval {i + 1}/{budget}")
        vector = rng.uniform(0.0, 1.0, benchmark.encoding_dim)
        score = benchmark.evaluate(vector)
        if best is None or score < best:
            best = score
    return best


def random_baseline_median(benchmark, budget, n_repeats, seed):
    rng = np.random.default_rng(seed)
    repeat_seeds = rng.integers(0, 2**31 - 1, size=n_repeats)
    bests = []
    for i, s in enumerate(repeat_seeds, 1):
        bests.append(random_search_best(benchmark, budget, int(s), desc=f"reference run {i}/{n_repeats}"))
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
