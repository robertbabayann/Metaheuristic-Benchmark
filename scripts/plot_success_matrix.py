import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import storage
from optimizers import ALGORITHM_POOL


def _load_matrix(run_dir, mode, metric_key):
    pattern = os.path.join(run_dir, "*", "*", "*.json")
    datasets = []
    rates = {}
    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)
        if data["mode"] != mode:
            continue
        track = path.split(os.sep)[-3]
        dataset = path.split(os.sep)[-2]
        column = f"{track}/{dataset}"
        if column not in datasets:
            datasets.append(column)
        records = data["records"]
        finite = [r for r in records if np.isfinite(r.get(metric_key, float("inf")))]
        rate = len(finite) / len(records) if records else 0.0
        rates[(data["algorithm"], column)] = rate

    algorithms = [name for name in ALGORITHM_POOL if any(name == alg for alg, _ in rates)]
    datasets = sorted(datasets)
    matrix = np.full((len(algorithms), len(datasets)), np.nan)
    for i, algo in enumerate(algorithms):
        for j, column in enumerate(datasets):
            if (algo, column) in rates:
                matrix[i, j] = rates[(algo, column)]
    return algorithms, datasets, matrix


def plot_matrix(algorithms, datasets, matrix, title, out_path):
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.1), max(6, len(algorithms) * 0.45)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    for i in range(len(algorithms)):
        for j in range(len(datasets)):
            value = matrix[i, j]
            text = "-" if np.isnan(value) else f"{value * 100:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="success rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--metric", default="target_2")
    args = parser.parse_args()

    run_id = args.run or (storage.list_runs()[0] if storage.list_runs() else None)
    if run_id is None:
        print("no runs found")
        return

    run_dir = storage.run_dir(run_id)
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for mode in ("default", "tuned"):
        algorithms, datasets, matrix = _load_matrix(run_dir, mode, args.metric)
        plot_matrix(algorithms, datasets, matrix, f"Success rate matrix — {mode}", os.path.join(out_dir, f"success_matrix_{mode}.png"))

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
