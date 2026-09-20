import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage
from ranking import build_ranking_table, format_ranking_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id (bench_...); defaults to the latest run")
    parser.add_argument("--metric", default="target_2")
    args = parser.parse_args()

    run_id = args.run or (storage.list_runs()[0] if storage.list_runs() else None)
    if run_id is None:
        print("no runs found")
        return

    rows = build_ranking_table(storage.run_dir(run_id), metric_key=args.metric)
    print(format_ranking_table(rows))


if __name__ == "__main__":
    main()
