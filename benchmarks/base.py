import json
import os
from abc import ABC, abstractmethod


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
