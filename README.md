# Metaheuristic Benchmark on Real ML Tasks

A benchmark comparing 22 metaheuristic optimizers (bio-inspired, swarm, physics-based and
classical) on two real machine-learning tasks, instead of synthetic test functions:

- **HPO** — hyperparameter optimization of a small fully-connected network, via the
  [FCNet tabular benchmark](https://arxiv.org/abs/1905.04970) (Klein & Hutter), on
  `protein_structure`, `slice_localization`, `naval_propulsion`, `parkinsons_telemonitoring`.
- **NAS** — neural architecture search on the topology search space of
  [NATS-Bench](https://github.com/D-X-Y/NATS-Bench), on `cifar10-valid`, `cifar100`,
  `ImageNet16-120`.

Both tasks are fully tabulated: every candidate's fitness is a lookup into a precomputed table
of real training results, not a live model training run and not a synthetic function. This makes
it feasible to run 22 optimizers, in `default` and `tuned` modes, with 5 seeds each, at a shared
query budget, and to know the true global optimum of the search space exactly.

## Methodology

- **Search space encoding.** Every optimizer works over a continuous vector in `[0, 1]^d`
  (mealpy's `FloatVar`), regardless of the underlying space's actual type. The HPO track decodes
  each coordinate to the nearest point on the benchmark's hyperparameter grid; the NAS track
  encodes each of the 6 cell edges as a one-hot block and decodes it via `argmax`. The same
  encoding is used for every algorithm, including the (real-coded) genetic algorithm, so no
  method gets a hand-tuned representation advantage.
- **Query budget.** `SEARCH_BUDGET` real benchmark lookups, identical for every algorithm on a
  given task. This is enforced *exactly*, not estimated — see "Exact query-budget accounting"
  below.
- **Targets.** `optimum` is the true global optimum (exhaustive search over the — fully or near-
  fully enumerable — space). `random_median` is the median best-of-run over several independent
  random-search runs, each with the *same* query budget as the algorithms being compared (a
  median over single random points would systematically flatter every algorithm and make the
  comparison uninformative). `target_1/2/3` sit at 50% / 75% / 90% of the way from
  `random_median` to `optimum`.
- **Default vs. tuned.** Each algorithm is run both with library-default internal parameters and
  with parameters found by TPE (Optuna) on a shorter proxy budget, per dataset. This answers "is
  tuning worth it for this method" separately from "which method wins outright."
- **Performance profiles.** Dolan–Moré profiles over `Queries_to_TargetX`, computed **per
  (dataset, seed)** rather than averaged across seeds first — see "Performance profiles no longer
  hide partial failures" below.

## Repository layout

```
config.py         constants: budgets, datasets, seeds, target levels
benchmarks.py      Benchmark base class, [0,1]^d decode helpers, FCNetBenchmark, NATSBenchmark
optimizers.py      the 22-algorithm pool, mealpy registry glue, the VCS bug patch
metrics.py         QueryBudget/BudgetExceeded, target calibration, queries-to-target,
                    Dolan-Moré performance profiles
tuning.py          TPE tuning of internal parameters (Optuna), budget-accurate
experiment.py      progress bar, benchmark construction, target caching, the experiment runner
run.py             interactive CLI
scripts/
  run_experiment.py            non-interactive: run the whole pool on one (track, dataset)
  build_performance_profile.py text summary of performance profiles
  plot_performance_profile.py  linear + log2 performance-profile plots
  inspect_results.py           raw per-(algorithm, dataset, mode) diagnostic dump
  check_fcnet_wrapper.py       one-off sanity check against the raw FCNet data (see below)
tests/
  test_core.py       decode helpers, pop_size sanitisation, exact-budget accounting (incl. the
                      previously-broken multi-phase algorithms), metrics
data/                benchmark data files (downloaded manually, see below — not in git)
cache/               global optima and targets, computed once and cached
results/             one JSON per (algorithm, dataset, mode)
```

Six modules instead of the previous seven sub-packages/eighteen files: each module groups
everything about one concern (all benchmark code together, all metrics together, etc.) rather
than one file per function.

## Installation

```bash
git clone <this repo>
cd <this repo>
pip install -e .
```

This also pulls in `tabular_benchmarks` (the FCNet reader) directly from its GitHub source
(`automl/nas_benchmarks`) as an ordinary pip dependency — no manual cloning or separate install
step. Note: importing `tabular_benchmarks` normally would fail, because its `__init__.py`
unconditionally imports a NAS-Bench-101 module that depends on the (non-pip, TensorFlow-based)
`nasbench` package, which this project doesn't need. `benchmarks.py` loads
`tabular_benchmarks/fcnet_benchmark.py` directly by file path (`importlib`), bypassing that
`__init__.py` entirely, so nothing extra needs to be installed.

## Data setup

The code does not — and cannot — bundle the benchmark data (several GB). Download it once:

**FCNet (HPO track):**
```bash
mkdir -p data/fcnet_tabular_benchmarks
wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
tar xf fcnet_tabular_benchmarks.tar.gz -C data/fcnet_tabular_benchmarks --strip-components=1
```

**NATS-Bench (NAS track):** the topology-search-space archive (`NATS-tss-v1_0-3ffb9-simple.tar`)
is distributed via Google Drive; follow the "Preparation and Download" section of the
[NATS-Bench README](https://github.com/D-X-Y/NATS-Bench#preparation-and-download) and extract it
to `data/NATS-tss-v1_0-3ffb9-simple` (or point `config.NATS_TSS_FILE` at wherever you put it).

After downloading, it's worth running the sanity check once:
```bash
python scripts/check_fcnet_wrapper.py protein_structure
```
It prints, for a few random configurations, the four raw stored training repeats next to this
project's averaged value, and asserts they're consistent. This wrapper was written and reasoned
about against the upstream library's source, but not run end-to-end against the real data before
your first run — worth the one-time check.

## Usage

Interactive:
```bash
python run.py
```

Non-interactive, one track/dataset, full pool:
```bash
python scripts/run_experiment.py --track hpo --dataset protein_structure
python scripts/run_experiment.py --track nas --dataset cifar100 --modes tuned --algorithms GA PSO VCS
```

Analysis, once you have results:
```bash
python scripts/build_performance_profile.py          # text table
python scripts/plot_performance_profile.py            # profile_linear.png / profile_log2.png
python scripts/inspect_results.py --metric target_2 --out diagnostics.csv
```

Tests:
```bash
pip install -e ".[dev]"
pytest tests/
```

## What changed vs. the previous version, and why

This is a rewrite that fixes several issues found during a code review, on top of the file
consolidation described above.

1. **Exact query-budget accounting (`metrics.QueryBudget`).** The previous implementation derived
   the number of real benchmark queries per epoch as `pop_size`, and computed the number of
   epochs as `budget // pop_size`. This is wrong for any algorithm whose `evolve()` step evaluates
   more than one full population per epoch — confirmed, by reading mealpy's source, for **VCS**
   (3 full-population phases per epoch), **AEO** (2 phases), **SSA** (~1.5–2×, depending on the
   tuned `SD` parameter), and **HHO** (a stochastic per-individual extra evaluation in its
   levy-flight branch). Those algorithms were silently consuming 1.5–3× more real benchmark
   queries than the shared budget was supposed to allow, while being logged and scored as if they
   hadn't — a direct violation of the "same query budget for everyone" fairness principle, and it
   inflated their apparent `Queries_to_TargetX` performance.

   The fix wraps every benchmark call in a real counter (`QueryBudget`) that raises once the
   budget is spent, regardless of *why* the underlying optimizer wanted another query. Optimizers
   are given a generous epoch cap (`epoch = budget`) and the counter — not the epoch count — is
   what actually stops the run. This is exact for every algorithm's internal structure, present
   and future, without needing a per-algorithm cost model. `queries_to_target` is now read
   directly off the per-query best-so-far trace the counter records, so it no longer depends on a
   `queries_per_epoch` estimate either. Verified in `tests/test_core.py` against real mealpy
   optimizers (including VCS, AEO, SSA, HHO): every one of them now consumes exactly `budget`
   queries, not more.

2. **Deterministic fitness.** Both underlying tables have a "randomly pick one repeat" default
   that is not controlled by this project's own seed policy: FCNet's `objective_function` draws
   `rng.randint(4)` from an unseeded RNG on every call, and NATS-Bench's `get_more_info` defaults
   to `is_random=True`. Left alone, this means the "true global optimum" is a single noisy draw,
   re-rolled every time the cache is rebuilt, rather than the fact the methodology section claims
   it is. NATS-Bench is fixed with the one-line `is_random=False` (library-supported: average over
   all recorded seeds). FCNet has no such flag, so `FCNetBenchmark` reads the four stored repeats
   directly off the same `tabular_benchmarks` object's loaded HDF5 data and averages them itself,
   instead of calling the library's randomized accessor.

3. **Performance profiles no longer hide partial failures.** The previous profile builder averaged
   `Queries_to_TargetX` over the 5 seeds *before* handing costs to the Dolan–Moré routine,
   counting a seed that never reached the target as if it simply weren't there — an algorithm
   that reached the target on 3 of 5 seeds and failed on 2 got the same cost as one that reached
   it reliably on all 5. `metrics.load_cost_table` now treats each `(dataset, seed)` pair as its
   own problem instance, so a failed seed shows up as its own unresolved point in the profile
   instead of being averaged away.

4. Removed the redundant, duplicated `load_cost_table` implementation that used to live in both
   plotting and text-summary scripts.

5. **Tuning no longer overfits to one random draw (`tuning.py`).** Every Optuna trial used to call
   `optimizer.solve(problem, seed=tuning_seed)` with the *same* fixed seed across all trials —
   meaning TPE wasn't picking the internal parameters that work best for the algorithm in general,
   it was picking whatever exploits that one specific random population trajectory best. This gets
   worse, not better, as `TUNING_N_TRIALS` grows, and is the most likely explanation for several
   algorithms scoring *lower* AUC in `tuned` mode than in `default` mode after `TUNING_N_TRIALS` was
   raised to 100. Fixed by deriving a distinct, deterministic seed per trial
   (`tuning_seed + trial.number`): each candidate configuration is now evaluated on its own random
   draw, while the overall tuning stage stays fully reproducible run-to-run.

6. Added `scipy` to `pyproject.toml` — required by the Friedman/Wilcoxon significance report in
   `ranking.py`, which wasn't declared as a dependency.

## A note on `cache/targets/`

Target calibration (`optimum`, `random_median`, `target_*`) is cached per dataset under
`cache/targets/<track>/<dataset>.json` and is **not** scoped to a `results/bench_<timestamp>/` run.
Changing any of `SEARCH_BUDGET`, `RANDOM_SEARCH_REPEATS`, or `TARGET_LEVELS` in `config.py` silently
invalidates every existing file there — they are not recomputed automatically. Run
`rm -rf cache/targets` after any such change and before starting a new run, or different datasets in
the same `results/bench_*` folder can end up calibrated against different, inconsistent budgets
without any error or warning. `cache/hpo_optimum/` and `cache/nas_optimum/` are unaffected by these
settings and do not need to be cleared.

## License

MIT — see [LICENSE](LICENSE). Change it if you'd rather use something else; nothing in the code
depends on the choice.