import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from benchmarks.hpo_fcnet import FCNetBenchmark
from benchmarks.nas_nats import NATSBenchmark
from experiment.runner import get_task_targets, run_algorithm_on_task
from optimizers.pool import ALGORITHM_POOL


def build_benchmark(track, dataset):
    if track == "hpo":
        cache_path = os.path.join(config.CACHE_DIR, "hpo_optimum", f"{dataset}.json")
        return FCNetBenchmark(
            dataset=dataset,
            data_dir=config.FCNET_DATA_DIR,
            budget=config.FCNET_TRAIN_BUDGET_EPOCHS,
            cache_path=cache_path,
        )
    if track == "nas":
        cache_path = os.path.join(config.CACHE_DIR, "nas_optimum", f"{dataset}.json")
        return NATSBenchmark(
            dataset=dataset,
            file_path=config.NATS_TSS_FILE,
            hp=config.NAS_HP,
            cache_path=cache_path,
        )
    raise ValueError(f"unknown track: {track}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["hpo", "nas"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["default", "tuned", "both"], default="both")
    parser.add_argument("--algorithms", default="all")
    args = parser.parse_args()

    benchmark = build_benchmark(args.track, args.dataset)

    targets_cache_path = os.path.join(config.CACHE_DIR, "targets", args.track, f"{args.dataset}.json")
    targets = get_task_targets(
        benchmark,
        targets_cache_path,
        config.RANDOM_BASELINE_SAMPLES,
        config.RANDOM_BASELINE_SEED,
        config.TARGET_LEVELS,
    )

    if args.algorithms == "all":
        algorithm_names = list(ALGORITHM_POOL.keys())
    else:
        algorithm_names = args.algorithms.split(",")

    modes = config.MODES if args.mode == "both" else (args.mode,)

    output_dir = os.path.join(config.RESULTS_DIR, args.track, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    for algorithm_name in algorithm_names:
        algorithm_spec = ALGORITHM_POOL[algorithm_name]
        for mode in modes:
            result = run_algorithm_on_task(
                algorithm_name,
                algorithm_spec,
                benchmark,
                targets,
                mode,
                config.SEARCH_BUDGET,
                config.FINAL_SEEDS,
                config.TUNING_SEED,
                config.TUNING_N_TRIALS,
                config.TUNING_PROXY_FRACTION,
            )
            output_path = os.path.join(output_dir, f"{algorithm_name}_{mode}.json")
            with open(output_path, "w") as f:
                json.dump({"targets": targets, **result}, f, indent=2)
            print(f"{args.track}/{args.dataset} {algorithm_name} {mode} -> {output_path}")


if __name__ == "__main__":
    main()
