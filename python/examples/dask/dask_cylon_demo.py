"""
Run
>> mpirun -n 4 python3 python/examples/dask_mpi_cylon/dask_cylon_demo.py
"""

from typing import List
from dask_mpi import initialize
import dask.dataframe as dd
from dask.distributed import Client, wait, futures_of
from pycylon import CylonContext
from pycylon.net import MPIConfig
from pycylon import Table
import numpy as np

initialize()
client = Client()

npartitions = 4

data_size = 1000
num_chunks = 10


def print_data_and_rank(chunks: list):
    """ Fake function that mocks out how an MPI function should operate

    -   It takes in a list of chunks of data that are present on this machine
    -   It does whatever it wants to with this data and MPI
        Here for simplicity we just print the data and print the rank
    -   Maybe it returns something
    """
    mpi_config = MPIConfig()
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    rank = ctx.get_rank()
    size = ctx.get_world_size()
    #assert chunks[0].shape[0] * len(chunks) * (size - 2) == data_size
    #print(len(chunks), chunks[0].shape)

    col_names = [f'col_{i}' for i in range(len(chunks))]
    cn_tb: Table = Table.from_numpy(ctx, col_names, chunks)
    cn_tbs: List[Table] = [cn_tb.sum(i) for i in range(len(chunks))]
    #cn_tb_all: Table = Table.merge(ctx, cn_tbs)
    #print(cn_tb.column_count, cn_tb.row_count, cn_tb_all.column_count, cn_tb_all.row_count)
    local_sum = sum(chunk.sum() for chunk in chunks)
    print(f">>>Rank[{rank}] Sum: {local_sum}")
    return local_sum


# Use Dask Array to "load" data (actually just create random data here)

import dask.array as da

x = da.random.random(data_size, chunks=(num_chunks,))
x = x.persist()
wait(x)

# df_l = dd.read_csv(f"/tmp/user_usage_tm_1.csv").repartition(npartitions=npartitions)
# df_r = dd.read_csv(f"/tmp/user_device_tm_1.csv").repartition(npartitions=npartitions)

# Find out where data is on each worker
# TODO: This could be improved on the Dask side to reduce boiler plate

from toolz import first
from collections import defaultdict

key_to_part_dict = {str(part.key): part for part in futures_of(x)}
who_has = client.who_has(x)
worker_map = defaultdict(list)
for key, workers in who_has.items():
    worker_map[first(workers)].append(key_to_part_dict[key])

# Call an MPI-enabled function on the list of data present on each worker

print(worker_map.keys())

futures = [client.submit(print_data_and_rank, list_of_parts, workers=worker)
           for worker, list_of_parts in worker_map.items()]

wait(futures)

client.close()