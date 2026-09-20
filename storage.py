import glob
import os
import time

import config


def new_run_id():
    return time.strftime("bench_%Y%m%d_%H%M%S")


def run_dir(run_id):
    return os.path.join(config.RESULTS_DIR, run_id)


def settings_path(run_id):
    return os.path.join(run_dir(run_id), "run_settings.json")


def list_runs():
    if not os.path.isdir(config.RESULTS_DIR):
        return []
    names = [
        name
        for name in os.listdir(config.RESULTS_DIR)
        if name.startswith("bench_") and os.path.isdir(os.path.join(config.RESULTS_DIR, name))
    ]
    return sorted(names, reverse=True)


def result_path(run_id, track, dataset, algorithm, mode):
    return os.path.join(run_dir(run_id), track, dataset, f"{algorithm}_{mode}.json")


def full_universe(tracks_datasets, algorithms, modes):
    combos = []
    for track, dataset in tracks_datasets:
        for algorithm in algorithms:
            for mode in modes:
                combos.append((track, dataset, algorithm, mode))
    return combos


def missing_combinations(run_id, tracks_datasets, algorithms, modes):
    return [
        combo
        for combo in full_universe(tracks_datasets, algorithms, modes)
        if not os.path.exists(result_path(run_id, *combo))
    ]


def run_progress(run_id, tracks_datasets, algorithms, modes):
    universe = full_universe(tracks_datasets, algorithms, modes)
    done = sum(1 for combo in universe if os.path.exists(result_path(run_id, *combo)))
    return done, len(universe)

