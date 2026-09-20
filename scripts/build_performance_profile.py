import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from metrics import load_cost_table, performance_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default=None, help="if omitted, uses target_1/target_2/target_3 together")
    parser.add_argument("--tau-max", type=float, default=32.0)
    parser.add_argument("--n-taus", type=int, default=9)
    args = parser.parse_args()

    metrics = [args.metric] if args.metric else ["target_1", "target_2", "target_3"]
    cost_table = load_cost_table(config.RESULTS_DIR, metrics)

    if not cost_table:
        print("no results found yet")
        return

    taus = np.geomspace(1.0, args.tau_max, args.n_taus)
    profiles = performance_profile(cost_table, taus)

    header = "method".ljust(16) + "".join(f"tau={t:<8.2g}" for t in taus)
    print(header)
    for method, curve in sorted(profiles.items()):
        row = method.ljust(16) + "".join(f"{v:<12.3f}" for v in curve)
        print(row)


if __name__ == "__main__":
    main()
