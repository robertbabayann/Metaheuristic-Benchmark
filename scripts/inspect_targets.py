import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def main():
    pattern = os.path.join(config.CACHE_DIR, "targets", "*", "*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no cached targets found under {os.path.join(config.CACHE_DIR, 'targets')}")
        return

    header = f"{'track/dataset':<28} {'optimum':>12} {'random_median':>14} {'gap':>12} {'gap_pct_of_median':>18}"
    print(header)
    print("-" * len(header))
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        track = os.path.basename(os.path.dirname(path))
        dataset = os.path.splitext(os.path.basename(path))[0]
        optimum = data["optimum"]
        median = data["random_median"]
        gap = median - optimum
        gap_pct = (gap / median * 100) if median else float("nan")
        print(f"{track + '/' + dataset:<28} {optimum:>12.6g} {median:>14.6g} {gap:>12.6g} {gap_pct:>17.4f}%")
        for key in sorted(k for k in data if k.startswith("target_")):
            print(f"    {key} = {data[key]:.6g}")


if __name__ == "__main__":
    main()