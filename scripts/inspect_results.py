import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config


def load_raw_table(metric_key):
    rows = []
    pattern = os.path.join(config.RESULTS_DIR, "*", "*", "*.json")
    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)

        track = path.split(os.sep)[-3]
        dataset = path.split(os.sep)[-2]
        values = [record[metric_key] for record in data["records"]]
        finite_values = [v for v in values if np.isfinite(v)]

        rows.append(
            {
                "track": track,
                "dataset": dataset,
                "algorithm": data["algorithm"],
                "mode": data["mode"],
                "n_seeds": len(values),
                "n_inf": len(values) - len(finite_values),
                "mean": float(np.mean(finite_values)) if finite_values else float("inf"),
                "min": float(np.min(finite_values)) if finite_values else float("inf"),
                "max": float(np.max(finite_values)) if finite_values else float("inf"),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="target_2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_raw_table(args.metric)
    rows.sort(key=lambda r: (r["track"], r["dataset"], r["algorithm"], r["mode"]))

    for r in rows:
        print(
            f"{r['track']}/{r['dataset']:24s} {r['algorithm']:8s} {r['mode']:8s} "
            f"mean={r['mean']:.1f} min={r['min']:.1f} max={r['max']:.1f} inf={r['n_inf']}/{r['n_seeds']}"
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
