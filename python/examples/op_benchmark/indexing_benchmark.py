import time
import pandas as pd
import numpy as np
import pycylon as cn
from pycylon import Table
from pycylon.io import CSVWriteOptions
from pycylon.indexing.index import IndexingSchema
import argparse
from bench_util import get_dataframe

"""
Run benchmark:

>>> python python/examples/op_benchmark/indexing_benchmark.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 20_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/indexing_bench.csv \
                                        --duplication_factor 0.9 \
                                        --repetitions 5
"""


def indexing_op(num_rows: int, num_cols: int, duplication_factor: float):
    ctx: cn.CylonContext = cn.CylonContext(config=None, distributed=False)
    pdf = get_dataframe(num_rows=num_rows, num_cols=num_cols, duplication_factor=duplication_factor)
    pdf1 = pdf.copy()
    filter_column = pdf.columns[0]
    filter_column_data = pdf[pdf.columns[0]]
    random_index = np.random.randint(low=0, high=pdf.shape[0])
    filter_value = filter_column_data.values[random_index]
    tb = Table.from_pandas(ctx, pdf)
    cylon_indexing_time = time.time()
    tb.set_index(filter_column, drop=True)
    cylon_indexing_time = time.time() - cylon_indexing_time
    pdf_indexing_time = time.time()
    pdf.set_index(filter_column, drop=True, inplace=True)
    pdf_indexing_time = time.time() - pdf_indexing_time
    pdf_eval_indexing_time = time.time()
    pdf1 = pd.eval("pdf1.set_index(filter_column, drop=True)")
    pdf_eval_indexing_time = time.time() - pdf_eval_indexing_time

    cylon_filter_time = time.time()
    tb_filter = tb.loc[filter_value]
    cylon_filter_time = time.time() - cylon_filter_time

    pandas_filter_time = time.time()
    pdf_filtered = pdf.loc[filter_value]
    pandas_filter_time = time.time() - pandas_filter_time

    pandas_eval_filter_time = time.time()
    pdf_filtered_eval = pd.eval("pdf1.loc[filter_value]")
    pandas_eval_filter_time = time.time() - pandas_eval_filter_time

    del pdf
    del pdf1
    del tb
    del tb_filter
    del pdf_filtered
    del pdf_filtered_eval

    return pandas_filter_time, cylon_filter_time, pandas_eval_filter_time, pdf_indexing_time, cylon_indexing_time, pdf_eval_indexing_time


def bench_indexing_op(start: int, end: int, step: int, num_cols: int, repetitions: int, stats_file: str,
                      duplication_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "pandas_loc", "cylon_loc", "pandas_eval_loc", "speed up loc",
              "speed up loc(eval)", "pandas_indexing", "cylon_indexing",
              "pdf_eval_indexing", "speed up indexing", "speed up indexing(eval)"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            pandas_filter_time, cylon_filter_time, pandas_eval_filter_time, pdf_indexing_time, cylon_indexing_time, pdf_eval_indexing_time = indexing_op(
                num_rows=records, num_cols=num_cols,
                duplication_factor=duplication_factor)
            times.append(
                [pandas_filter_time, cylon_filter_time, pandas_eval_filter_time, pdf_indexing_time, cylon_indexing_time,
                 pdf_eval_indexing_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(
            f"Loc Op : Records={records}, Columns={num_cols}, Pandas Loc Time : {times[0]}, "
            f"Cylon Loc Time : {times[1]}, "
            f"Pandas Indexing Time : {times[2]}, Cylon Indexing Time : {times[3]}")
        all_data.append(
            [records, num_cols, times[0], times[1], times[2], times[0] / times[1], times[2] / times[1], times[3],
             times[4], times[5], times[3] / times[4], times[5] / times[4]])
    pdf = pd.DataFrame(all_data, columns=schema)
    print(pdf)
    pdf.to_csv(stats_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--start_size",
                        help="initial data size",
                        type=int)
    parser.add_argument("-e", "--end_size",
                        help="end data size",
                        type=int)
    parser.add_argument("-d", "--duplication_factor",
                        help="random data duplication factor",
                        type=float)
    parser.add_argument("-s", "--step_size",
                        help="Step size",
                        type=int)
    parser.add_argument("-c", "--num_cols",
                        help="number of columns",
                        type=int)
    parser.add_argument("-t", "--filter_size",
                        help="number of values per filter",
                        type=int)
    parser.add_argument("-r", "--repetitions",
                        help="number of experiments to be repeated",
                        type=int)
    parser.add_argument("-f", "--stats_file",
                        help="stats file to be saved",
                        type=str)

    args = parser.parse_args()
    print(f"Start Data Size : {args.start_size}")
    print(f"End Data Size : {args.end_size}")
    print(f"Step Data Size : {args.step_size}")
    print(f"Data Duplication Factor : {args.duplication_factor}")
    print(f"Number of Columns : {args.num_cols}")
    print(f"Number of Repetitions : {args.repetitions}")
    print(f"Stats File : {args.stats_file}")
    bench_indexing_op(start=args.start_size,
                      end=args.end_size,
                      step=args.step_size,
                      num_cols=args.num_cols,
                      repetitions=args.repetitions,
                      stats_file=args.stats_file,
                      duplication_factor=args.duplication_factor)
