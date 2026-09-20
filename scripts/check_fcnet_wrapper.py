import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from benchmarks import FCNetBenchmark


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else config.HPO_DATASETS[0]
    bench = FCNetBenchmark(dataset=dataset, data_dir=config.FCNET_DATA_DIR, budget=config.FCNET_TRAIN_BUDGET_EPOCHS)

    rng = np.random.default_rng(0)
    print(f"dataset={dataset}  hyperparameters={bench.hp_names}")
    for _ in range(5):
        vector = rng.uniform(0.0, 1.0, bench.encoding_dim)
        values = bench.decode(vector)
        key_runs = bench.bench.data[
            __import__("json").dumps(values, sort_keys=True)
        ]["valid_mse"][:, bench.budget - 1]
        our_value = bench.evaluate(vector)
        print(f"config={values}")
        print(f"  raw 4 repeats = {list(key_runs)}")
        print(f"  our average   = {our_value}")
        assert abs(our_value - float(np.mean(key_runs))) < 1e-12


if __name__ == "__main__":
    main()
