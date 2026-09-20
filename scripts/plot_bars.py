import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import storage
from ranking import build_ranking_table


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
    rows = build_ranking_table(run_dir, metric_key=args.metric)
    if not rows:
        print("no results found yet")
        return

    rows = sorted(rows, key=lambda r: r["best_auc"], reverse=True)
    names = [r["algorithm"] for r in rows]
    auc_default = [r.get("auc_default", 0.0) for r in rows]
    auc_tuned = [r.get("auc_tuned", 0.0) for r in rows]
    succ_default = [r.get("success_rate_default", 0.0) * 100 for r in rows]
    succ_tuned = [r.get("success_rate_tuned", 0.0) * 100 for r in rows]

    x = np.arange(len(names))
    width = 0.38

    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.5), 6))
    ax.bar(x - width / 2, auc_default, width, label="Default", color="#4C72B0")
    ax.bar(x + width / 2, auc_tuned, width, label="Tuned", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Performance-profile AUC")
    ax.set_title("Default vs Tuned — overall AUC")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bars_auc.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.5), 6))
    ax.bar(x - width / 2, succ_default, width, label="Default", color="#4C72B0")
    ax.bar(x + width / 2, succ_tuned, width, label="Tuned", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(f"Success rate on {args.metric}, %")
    ax.set_title("Default vs Tuned — success rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bars_success_rate.png"), dpi=150)
    plt.close(fig)

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
