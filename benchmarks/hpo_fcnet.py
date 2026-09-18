import itertools

from ConfigSpace import Configuration

from benchmarks.base import Benchmark
from encoding.ordinal_grid import nearest_choice_index

_CLASS_NAMES = {
    "protein_structure": "FCNetProteinStructureBenchmark",
    "slice_localization": "FCNetSliceLocalizationBenchmark",
    "naval_propulsion": "FCNetNavalPropulsionBenchmark",
    "parkinsons_telemonitoring": "FCNetParkinsonsTelemonitoringBenchmark",
}


def _choices(hp):
    return list(hp.choices) if hasattr(hp, "choices") else list(hp.sequence)


class FCNetBenchmark(Benchmark):
    def __init__(self, dataset, data_dir, budget, cache_path=None):
        super().__init__(cache_path=cache_path)
        import tabular_benchmarks

        benchmark_cls = getattr(tabular_benchmarks, _CLASS_NAMES[dataset])
        self.bench = benchmark_cls(data_dir=data_dir)
        self.cs = self.bench.get_configuration_space()
        self.hp_names = sorted(self.cs.keys())
        self.hp_choices = {name: _choices(self.cs[name]) for name in self.hp_names}
        self.budget = budget

    @property
    def encoding_dim(self):
        return len(self.hp_names)

    def decode(self, vector):
        values = {}
        for i, name in enumerate(self.hp_names):
            choices = self.hp_choices[name]
            idx = nearest_choice_index(vector[i], len(choices))
            values[name] = choices[idx]
        return Configuration(self.cs, values=values)

    def evaluate(self, vector):
        config = self.decode(vector)
        y, _ = self.bench.objective_function(config, budget=self.budget)
        return float(y)

    def _compute_global_optimum(self):
        best = None
        for combo in itertools.product(*(self.hp_choices[name] for name in self.hp_names)):
            values = dict(zip(self.hp_names, combo))
            config = Configuration(self.cs, values=values)
            y, _ = self.bench.objective_function(config, budget=self.budget)
            if best is None or y < best:
                best = y
        return float(best)
