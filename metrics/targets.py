import numpy as np

from experiment.progress import ProgressTracker


def random_baseline_median(benchmark, n_samples, seed):
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(n_samples):
        ProgressTracker.desc(f"reference points {i + 1}/{n_samples}")
        vector = rng.uniform(0.0, 1.0, benchmark.encoding_dim)
        scores.append(benchmark.evaluate(vector))
    return float(np.median(scores))


def compute_targets(benchmark, random_median, target_levels):
    optimum = benchmark.global_optimum()
    gap = random_median - optimum
    targets = {}
    for i, level in enumerate(target_levels, start=1):
        targets[f"target_{i}"] = random_median - level * gap
    targets["optimum"] = optimum
    targets["random_median"] = random_median
    return targets
