import sys

from tqdm import tqdm


class ProgressTracker:
    _bar = None

    @classmethod
    def phase(cls, total, desc):
        cls.close()
        if not total:
            return
        cls._bar = tqdm(
            total=int(total),
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            unit="q",
            unit_scale=True,
        )

    @classmethod
    def desc(cls, text):
        if cls._bar is not None:
            cls._bar.set_description(text)

    @classmethod
    def step(cls):
        bar = cls._bar
        if bar is None:
            return
        bar.update(1)
        if bar.n > bar.total:
            bar.total = bar.n

    @classmethod
    def close(cls):
        if cls._bar is None:
            return
        cls._bar.close()
        cls._bar = None
        sys.stdout.write("\n")
        sys.stdout.flush()

    @classmethod
    def write(cls, message):
        if cls._bar is not None:
            tqdm.write(message)
        else:
            print(message)


def track_progress(benchmark):
    if hasattr(benchmark, "_progress_wrapped"):
        return benchmark
    original = benchmark.evaluate

    def evaluate(vector):
        value = original(vector)
        ProgressTracker.step()
        return value

    benchmark.evaluate = evaluate
    benchmark._progress_wrapped = True
    return benchmark