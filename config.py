import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

FCNET_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "fcnet_tabular_benchmarks")
NATS_TSS_FILE = os.path.join(PROJECT_ROOT, "data", "NATS-tss-v1_0-3ffb9-simple")

HPO_DATASETS = [
    "protein_structure",
    "slice_localization",
    "naval_propulsion",
    "parkinsons_telemonitoring",
]

NAS_DATASETS = [
    "cifar10-valid",
    "cifar100",
    "ImageNet16-120",
]

NAS_HP = "200"

FCNET_TRAIN_BUDGET_EPOCHS = 100

SEARCH_BUDGET = 1200

FINAL_SEEDS = [1, 2, 3, 4, 5]
TUNING_SEED = 100

TUNING_N_TRIALS = 30
TUNING_PROXY_FRACTION = 0.2

RANDOM_SEARCH_REPEATS = 5
RANDOM_BASELINE_SEED = 777

TARGET_LEVELS = (0.50, 0.75, 0.90)

MODES = ("default", "tuned")
