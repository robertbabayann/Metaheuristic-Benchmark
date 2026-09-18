from benchmarks.base import Benchmark
from encoding.onehot_categorical import onehot_argmax_decode

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
        info = self.api.get_more_info(index, self.dataset, hp=self.hp)
        accuracy = info.get("valid-accuracy", info.get("test-accuracy"))
        score = 1.0 - accuracy / 100.0
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
