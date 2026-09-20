import importlib.util
import itertools
import json
import os
from abc import ABC, abstractmethod

import numpy as np


def nearest_choice_index(value, n_choices):
    clipped = min(max(value, 0.0), 1.0)
    idx = int(clipped * n_choices)
    return min(idx, n_choices - 1)


def onehot_argmax_decode(vector, n_slots, n_choices):
    matrix = np.asarray(vector).reshape(n_slots, n_choices)
    return matrix.argmax(axis=1)


class Benchmark(ABC):
    def __init__(self, cache_path=None):
        self.cache_path = cache_path
        self._optimum_cache = None

    @property
    @abstractmethod
    def encoding_dim(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, vector):
        raise NotImplementedError

    @abstractmethod
    def _compute_global_optimum(self):
        raise NotImplementedError

    def global_optimum(self):
        if self._optimum_cache is not None:
            return self._optimum_cache
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                cached = json.load(f)
            self._optimum_cache = cached["global_optimum"]
            return self._optimum_cache
        value = self._compute_global_optimum()
        self._optimum_cache = value
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump({"global_optimum": value}, f)
        return value


FCNET_BENCHMARK_CLASSES = {
    "protein_structure": "FCNetProteinStructureBenchmark",
    "slice_localization": "FCNetSliceLocalizationBenchmark",
    "naval_propulsion": "FCNetNavalPropulsionBenchmark",
    "parkinsons_telemonitoring": "FCNetParkinsonsTelemonitoringBenchmark",
}


def _load_fcnet_benchmark_module():
    spec = importlib.util.find_spec("tabular_benchmarks")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "tabular_benchmarks is not installed. It is a pip dependency pulled "
            "straight from GitHub, so `pip install -e .` (see pyproject.toml) "
            "should fetch it automatically."
        )
    package_dir = list(spec.submodule_search_locations)[0]
    file_path = os.path.join(package_dir, "fcnet_benchmark.py")
    module_spec = importlib.util.spec_from_file_location("tabular_benchmarks_fcnet", file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _hp_choices(hyperparameter):
    if hasattr(hyperparameter, "sequence"):
        return list(hyperparameter.sequence)
    return list(hyperparameter.choices)


class FCNetBenchmark(Benchmark):
    def __init__(self, dataset, data_dir, budget, cache_path=None):
        super().__init__(cache_path=cache_path)
        module = _load_fcnet_benchmark_module()
        benchmark_cls = getattr(module, FCNET_BENCHMARK_CLASSES[dataset])
        self.bench = benchmark_cls(data_dir=data_dir)
        self.budget = budget

        cs = self.bench.get_configuration_space()
        self.hp_names = sorted(h.name for h in cs.get_hyperparameters())
        self.hp_choices = {name: _hp_choices(cs.get_hyperparameter(name)) for name in self.hp_names}

    @property
    def encoding_dim(self):
        return len(self.hp_names)

    def decode(self, vector):
        values = {}
        for i, name in enumerate(self.hp_names):
            choices = self.hp_choices[name]
            idx = nearest_choice_index(vector[i], len(choices))
            values[name] = choices[idx]
        return values

    def _score(self, values):
        key = json.dumps(values, sort_keys=True)
        runs = self.bench.data[key]["valid_mse"][:, self.budget - 1]
        return float(np.mean(runs))

    def evaluate(self, vector):
        return self._score(self.decode(vector))

    def _compute_global_optimum(self):
        best = None
        choice_lists = [self.hp_choices[name] for name in self.hp_names]
        for combo in itertools.product(*choice_lists):
            values = dict(zip(self.hp_names, combo))
            score = self._score(values)
            if best is None or score < best:
                best = score
        return float(best)


OPERATIONS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
NUM_EDGES = 6
NUM_ARCHITECTURES = 15625
ARCH_TEMPLATE = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|"


class NATSBenchmark(Benchmark):
    def __init__(self, dataset, file_path, hp="200", cache_path=None):
        super().__init__(cache_path=cache_path)
        from nats_bench import create

        self.api = create(file_path, "tss", fast_mode=True, verbose=False)
        self.dataset = dataset
        self.hp = hp
        self._score_cache = {}

    @property
    def encoding_dim(self):
        return NUM_EDGES * len(OPERATIONS)

    def _decode_arch_str(self, vector):
        choice_idx = onehot_argmax_decode(vector, NUM_EDGES, len(OPERATIONS))
        ops = [OPERATIONS[i] for i in choice_idx]
        return ARCH_TEMPLATE.format(*ops)

    def _score_by_index(self, index):
        if index in self._score_cache:
            return self._score_cache[index]
        info = self.api.get_more_info(index, self.dataset, hp=self.hp, is_random=False)
        score = 1.0 - info["valid-accuracy"] / 100.0
        self._score_cache[index] = score
        return score

    def evaluate(self, vector):
        arch_str = self._decode_arch_str(vector)
        index = self.api.query_index_by_arch(arch_str)
        return self._score_by_index(index)

    def _compute_global_optimum(self):
        best = None
        for index in range(NUM_ARCHITECTURES):
            score = self._score_by_index(index)
            if best is None or score < best:
                best = score
        return best
