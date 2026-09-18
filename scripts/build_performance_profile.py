import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from metrics.performance_profile import performance_profile


def load_cost_table(metric_key):
    cost_table = {}
    pattern = os.path.join(config.RESULTS_DIR, "*", "*", "*.json")
    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)

        algorithm = data["algorithm"]
        mode = data["mode"]
        method_key = f"{algorithm}_{mode}"

        track = path.split(os.sep)[-3]
        dataset = path.split(os.sep)[-2]
        problem_key = f"{track}/{dataset}"

        values = [record[metric_key] for record in data["records"]]
        finite_values = [v for v in values if np.isfinite(v)]
        cost = float(np.mean(finite_values)) if finite_values else float("inf")

        cost_table.setdefault(method_key, {})[problem_key] = cost

    return cost_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="target_2")
    parser.add_argument("--taus", default="1,2,4,8,16,32")
    args = parser.parse_args()

    cost_table = load_cost_table(args.metric)
    taus = [float(t) for t in args.taus.split(",")]
    profiles = performance_profile(cost_table, taus)

    for method, curve in profiles.items():
        print(method, [round(v, 3) for v in curve])


if __name__ == "__main__":
    main()
