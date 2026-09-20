import dataclasses
import json
import os

import config
import storage
from experiment import ProgressTracker, build_benchmark, get_task_targets, run_algorithm_on_task, track_progress
from optimizers import ALGORITHM_POOL

TRUE_WORDS = {"y", "yes", "true", "1"}
FALSE_WORDS = {"n", "no", "false", "0"}

ALL_ALGORITHMS = list(ALGORITHM_POOL.keys())
ALL_TRACKS_DATASETS = [("hpo", d) for d in config.HPO_DATASETS] + [("nas", d) for d in config.NAS_DATASETS]
ALL_MODES = ("default", "tuned")


@dataclasses.dataclass
class RunSettings:
    search_budget: int
    final_seeds: tuple
    tuning_seed: int
    tuning_n_trials: int
    tuning_proxy_fraction: float
    random_search_repeats: int
    random_baseline_seed: int


def default_settings():
    return RunSettings(
        search_budget=config.SEARCH_BUDGET,
        final_seeds=tuple(config.FINAL_SEEDS),
        tuning_seed=config.TUNING_SEED,
        tuning_n_trials=config.TUNING_N_TRIALS,
        tuning_proxy_fraction=config.TUNING_PROXY_FRACTION,
        random_search_repeats=config.RANDOM_SEARCH_REPEATS,
        random_baseline_seed=config.RANDOM_BASELINE_SEED,
    )


def parse_value(current, raw):
    if isinstance(current, bool):
        lowered = raw.lower()
        if lowered in TRUE_WORDS:
            return True
        if lowered in FALSE_WORDS:
            return False
        raise ValueError(raw)
    if isinstance(current, tuple):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        parsed = []
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError:
                parsed.append(part)
        return tuple(parsed)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def edit_settings(settings):
    print("Press Enter to keep the current value")
    for f in dataclasses.fields(settings):
        current = getattr(settings, f.name)
        while True:
            raw = input(f"  {f.name} [{current}]: ").strip()
            if not raw:
                break
            try:
                setattr(settings, f.name, parse_value(current, raw))
                break
            except (ValueError, json.JSONDecodeError):
                print("    invalid value, try again")


def parse_indices(raw, limit):
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(part)
        value = int(part)
        if not 1 <= value <= limit:
            raise ValueError(part)
        indices.append(value)
    return indices


def task_select_menu(title, options, all_label):
    full_index = len(options) + 1
    print(f"\n{title}")
    for index, name in enumerate(options, 1):
        print(f"  {index}) {name}")
    print(f"  {full_index}) {all_label}")
    print("  0) Back")

    raw = input(f"Select (e.g. 1,3 or {full_index}): ").strip()
    if raw == "0":
        return None
    try:
        indices = parse_indices(raw, full_index)
    except ValueError:
        print("Invalid choice")
        return None
    if full_index in indices:
        return list(options)
    return [options[index - 1] for index in indices]


def choose_one(title, options, prompt="Select"):
    print(f"\n{title}")
    for index, name in enumerate(options, 1):
        print(f"  {index}) {name}")
    print("  0) Back")

    raw = input(f"{prompt}: ").strip()
    if raw == "0":
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= len(options):
        print("Invalid choice")
        return None
    return options[int(raw) - 1]


def _custom_plan():
    tracks = task_select_menu("Tracks:", ["hpo", "nas"], "both tracks")
    if not tracks:
        return None
    tracks_datasets = [td for td in ALL_TRACKS_DATASETS if td[0] in tracks]

    dataset_names = [d for t, d in tracks_datasets]
    picked_datasets = task_select_menu("Datasets:", dataset_names, "all selected datasets")
    if picked_datasets is None:
        return None
    tracks_datasets = [td for td in tracks_datasets if td[1] in picked_datasets]

    modes = task_select_menu("Modes:", list(ALL_MODES), "both modes")
    if not modes:
        return None

    algorithms = task_select_menu("Algorithms:", ALL_ALGORITHMS, "all algorithms")
    if not algorithms:
        return None

    settings = default_settings()
    edit_settings(settings)
    return tracks_datasets, algorithms, modes, settings


def _prompt_use_defaults():
    raw = input("\nUse default settings? [Enter = yes, full run / n = customize]: ").strip().lower()
    return raw == "" or raw not in FALSE_WORDS


def _run_batch(run_id, combos, settings):
    if not combos:
        print("Nothing to run — this configuration is already complete.")
        return

    by_task = {}
    for track, dataset, algorithm, mode in combos:
        by_task.setdefault((track, dataset), []).append((algorithm, mode))

    total_budget = 0
    for (track, dataset), pairs in by_task.items():
        for algorithm, mode in pairs:
            proxy_budget = max(1, int(settings.search_budget * settings.tuning_proxy_fraction))
            tuning_budget = settings.tuning_n_trials * proxy_budget if mode == "tuned" else 0
            total_budget += settings.search_budget * len(settings.final_seeds) + tuning_budget

    ProgressTracker.start(total_budget, desc=run_id)

    try:
        for (track, dataset), pairs in by_task.items():
            task_label = f"{track}/{dataset}"
            benchmark = build_benchmark(track, dataset)
            track_progress(benchmark)

            targets_cache_path = os.path.join(config.CACHE_DIR, "targets", track, f"{dataset}.json")
            targets = get_task_targets(
                benchmark,
                targets_cache_path,
                settings.search_budget,
                settings.random_search_repeats,
                settings.random_baseline_seed,
                config.TARGET_LEVELS,
            )

            output_dir = os.path.join(storage.run_dir(run_id), track, dataset)
            os.makedirs(output_dir, exist_ok=True)

            for algorithm, mode in pairs:
                started = __import__("datetime").datetime.now().strftime("%H:%M:%S")
                result = run_algorithm_on_task(
                    algorithm,
                    ALGORITHM_POOL[algorithm],
                    benchmark,
                    targets,
                    mode,
                    settings.search_budget,
                    settings.final_seeds,
                    settings.tuning_seed,
                    settings.tuning_n_trials,
                    settings.tuning_proxy_fraction,
                    task_label=task_label,
                )
                output_path = storage.result_path(run_id, track, dataset, algorithm, mode)
                with open(output_path, "w") as f:
                    json.dump({"targets": targets, **result}, f, indent=2)

                fits = [r["final_fitness"] for r in result["records"]]
                hits = sum(1 for r in result["records"] if r.get("target_2") != float("inf"))
                mean_fit = sum(f for f in fits if f != float("inf")) / max(1, len(fits))
                ProgressTracker.log(
                    f"started {started} | {task_label} {algorithm} {mode} | "
                    f"hit target_2 {hits}/{len(result['records'])} | mean_fit={mean_fit:.4g}"
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        ProgressTracker.close()


def run_benchmark_menu():
    choice = choose_one("Run Benchmark:", ["New run", "Load existing run"])
    if choice is None:
        return

    if choice == "New run":
        use_defaults = _prompt_use_defaults()
        if use_defaults:
            tracks_datasets, algorithms, modes, settings = ALL_TRACKS_DATASETS, ALL_ALGORITHMS, list(ALL_MODES), default_settings()
        else:
            plan = _custom_plan()
            if plan is None:
                return
            tracks_datasets, algorithms, modes, settings = plan

        run_id = storage.new_run_id()
        os.makedirs(storage.run_dir(run_id), exist_ok=True)
        with open(storage.settings_path(run_id), "w") as f:
            json.dump(dataclasses.asdict(settings), f, indent=2)

        combos = storage.full_universe(tracks_datasets, algorithms, modes)
        print(f"\nStarting {run_id}: {len(combos)} runs ({len(tracks_datasets)} datasets x {len(algorithms)} algorithms x {len(modes)} modes)")
        _run_batch(run_id, combos, settings)
        return

    runs = storage.list_runs()
    if not runs:
        print("No existing runs found.")
        return
    labels = []
    for run_id in runs:
        done, total = storage.run_progress(run_id, ALL_TRACKS_DATASETS, ALL_ALGORITHMS, ALL_MODES)
        labels.append(f"{run_id} ({done}/{total})")
    picked = choose_one("Existing runs:", labels)
    if picked is None:
        return
    run_id = runs[labels.index(picked)]

    if os.path.exists(storage.settings_path(run_id)):
        with open(storage.settings_path(run_id), "r") as f:
            settings = RunSettings(**json.load(f))
        settings.final_seeds = tuple(settings.final_seeds)
    else:
        print(f"[warn] {run_id} has no saved settings, falling back to current defaults")
        settings = default_settings()

    missing = storage.missing_combinations(run_id, ALL_TRACKS_DATASETS, ALL_ALGORITHMS, ALL_MODES)
    if not missing:
        print(f"{run_id} is already complete.")
        return

    missing_datasets = sorted({f"{t}/{d}" for t, d, _, _ in missing})
    print(f"\n{run_id}: missing {len(missing)} runs across {len(missing_datasets)} datasets")

    use_defaults = _prompt_use_defaults()
    if use_defaults:
        combos = missing
    else:
        tracks = task_select_menu("Tracks to fill in:", ["hpo", "nas"], "both tracks")
        if not tracks:
            return
        algorithms = task_select_menu("Algorithms to fill in:", ALL_ALGORITHMS, "all missing algorithms")
        if algorithms is None:
            return
        modes = task_select_menu("Modes to fill in:", list(ALL_MODES), "both modes")
        if not modes:
            return
        combos = [c for c in missing if c[0] in tracks and c[2] in algorithms and c[3] in modes]

    _run_batch(run_id, combos, settings)


def _pick_run():
    runs = storage.list_runs()
    if not runs:
        print("No runs found.")
        return None
    labels = []
    for run_id in runs:
        done, total = storage.run_progress(run_id, ALL_TRACKS_DATASETS, ALL_ALGORITHMS, ALL_MODES)
        labels.append(f"{run_id} ({done}/{total})")
    picked = choose_one("Select run:", labels)
    if picked is None:
        return None
    return runs[labels.index(picked)]


def analysis_menu():
    run_id = _pick_run()
    if run_id is None:
        return
    from ranking import build_ranking_table, format_ranking_table

    rows = build_ranking_table(storage.run_dir(run_id))
    print()
    print(format_ranking_table(rows))


def plots_menu():
    run_id = _pick_run()
    if run_id is None:
        return
    import subprocess
    import sys as _sys

    for script in ("plot_performance_profile.py", "plot_bars.py", "plot_success_matrix.py"):
        subprocess.run([_sys.executable, os.path.join("scripts", script), "--run", run_id], check=False)


def show_menu():
    print("\nmeta-benchmark:")
    print("  1) Run Benchmark")
    print("  2) Analysis")
    print("  3) Build Plots")
    print("  0) Exit")


def main():
    while True:
        show_menu()
        raw = input("Select: ").strip()
        if raw == "0":
            break
        if raw == "1":
            run_benchmark_menu()
        elif raw == "2":
            analysis_menu()
        elif raw == "3":
            plots_menu()


if __name__ == "__main__":
    main()
