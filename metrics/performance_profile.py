import numpy as np


def performance_profile(cost_table, taus):
    methods = list(cost_table.keys())
    problems = list(next(iter(cost_table.values())).keys())

    ratios = {method: [] for method in methods}
    for problem in problems:
        values = {method: cost_table[method][problem] for method in methods}
        finite_values = [v for v in values.values() if np.isfinite(v)]
        if not finite_values:
            continue
        best = min(finite_values)
        for method in methods:
            v = values[method]
            ratios[method].append(v / best if np.isfinite(v) else np.inf)

    profiles = {}
    for method in methods:
        r = np.array(ratios[method])
        profiles[method] = np.array([np.mean(r <= tau) for tau in taus])
    return profiles
