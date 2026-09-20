import datetime
import json
import os
import time

from mealpy import FloatVar
from tqdm import tqdm

import config
from benchmarks import FCNetBenchmark, NATSBenchmark
from metrics import BudgetExceeded, QueryBudget, compute_targets, queries_to_targets, random_baseline_median
from optimizers import build_optimizer, default_pop_size, fit_pop_size, sanitize_pop_size
from tuning import tune_internal_params


class ProgressTracker:
    _bar = None

    @classmethod
    def start(cls, total, desc=""):
        cls.close()
        cls._bar = tqdm(total=int(total), desc=desc, dynamic_ncols=True, unit="q", unit_scale=True, leave=True)

    @classmethod
    def set_context(cls, text):
        if cls._bar is not None:
            cls._bar.set_description(text)

    @classmethod
    def step(cls, n=1):
        if cls._bar is None:
            return
        cls._bar.update(n)
        if cls._bar.n > cls._bar.total:
            cls._bar.total = cls._bar.n

    @classmethod
    def close(cls):
        if cls._bar is not None:
            cls._bar.close()
            cls._bar = None

    @classmethod
    def log(cls, message):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        if cls._bar is not None:
            tqdm.write(line)
        else:
            print(line)


def track_progress(benchmark):
    if getattr(benchmark, "_progress_wrapped", False):
        return benchmark
    original = benchmark.evaluate

    def evaluate(vector):
        value = original(vector)
        ProgressTracker.step()
        return value

    benchmark.evaluate = evaluate
    benchmark._progress_wrapped = True
    return benchmark


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


def get_task_targets(benchmark, cache_path, random_search_budget, random_search_repeats, random_baseline_seed, target_levels):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    median = random_baseline_median(benchmark, random_search_budget, random_search_repeats, random_baseline_seed)
    targets = compute_targets(benchmark, median, target_levels)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(targets, f)

    return targets


def _run_single_seed(class_name, params, pop_size, benchmark, budget, seed):
    tracker = QueryBudget(benchmark, budget)
    problem = {
        "obj_func": tracker,
        "bounds": FloatVar(lb=[0.0] * benchmark.encoding_dim, ub=[1.0] * benchmark.encoding_dim),
        "minmax": "min",
        "log_to": None,
    }
    try:
        optimizer = build_optimizer(class_name, budget, pop_size, params)
        optimizer.solve(problem, seed=seed)
    except BudgetExceeded:
        pass
    except Exception as exc:
        ProgressTracker.log(f"[warn] {class_name} seed={seed} raised {exc!r}; keeping partial history ({tracker.count} queries)")
    return tracker.best_history, tracker.count


def run_algorithm_on_task(
    algorithm_name,
    algorithm_spec,
    benchmark,
    targets,
    mode,
    search_budget,
    final_seeds,
    tuning_seed,
    tuning_n_trials,
    tuning_proxy_fraction,
    task_label=None,
):
    class_name = algorithm_spec["class_name"]
    defaults = algorithm_spec.get("defaults", {})

    context = f"{task_label} | {algorithm_name} | {mode}" if task_label else f"{algorithm_name} | {mode}"
    ProgressTracker.set_context(context)

    proxy_budget = max(1, int(search_budget * tuning_proxy_fraction))
    tuning_budget = tuning_n_trials * proxy_budget

    tuned_params = {}
    tuning_queries = 0
    tuning_wall_clock = 0.0

    if mode == "tuned":
        t0 = time.perf_counter()
        tuned_params, tuning_queries = tune_internal_params(
            class_name, algorithm_spec, benchmark, proxy_budget, tuning_n_trials, tuning_seed,
        )
        tuning_wall_clock = time.perf_counter() - t0
        best_params = {**defaults, **tuned_params}
    else:
        best_params = dict(defaults)

    internal_params = dict(best_params)
    if "pop_size" in internal_params:
        pop_size = int(internal_params.pop("pop_size"))
    else:
        pop_size = default_pop_size(class_name)
    pop_size = fit_pop_size(algorithm_spec, pop_size, search_budget)
    pop_size = sanitize_pop_size(algorithm_spec, pop_size, internal_params)

    records = []
    for seed in final_seeds:
        t0 = time.perf_counter()
        best_history, search_queries = _run_single_seed(class_name, internal_params, pop_size, benchmark, search_budget, seed)
        wall_clock_seconds = time.perf_counter() - t0
        q = queries_to_targets(best_history, targets)
        final_fitness = best_history[-1] if best_history else float("inf")
        records.append(
            {
                "seed": seed,
                "pop_size": pop_size,
                "final_fitness": final_fitness,
                "wall_clock_seconds": wall_clock_seconds,
                "search_budget": search_budget,
                "search_queries": search_queries,
                "tuning_budget": tuning_budget,
                "tuning_queries": tuning_queries,
                "total_budget": search_budget + tuning_budget,
                "total_queries": search_queries + tuning_queries,
                **q,
            }
        )

    return {
        "algorithm": algorithm_name,
        "mode": mode,
        "params": {**internal_params, "pop_size": pop_size},
        "tuned_params": tuned_params,
        "tuning_wall_clock_seconds": tuning_wall_clock,
        "records": records,
    }
