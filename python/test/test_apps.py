from pycylon import CylonContext
from pycylon import Table
import numpy as np
import pandas as pd
from pycylon.data.aggregates import AggregationOp

ctx = CylonContext(config=None, distributed=False)


def test_app_aggregate():
    pdf = pd.DataFrame([np.nan] * 5, dtype='object')
    target = 'AUC'
    records = 10

    a = np.random.randint(5, records)

    tb = Table.from_pydict(ctx, {'Source': np.random.randint(5, size=records).tolist(),
                                 'Drug1': np.random.randint(5, size=records).tolist(),
                                 'Drug2': np.random.randint(5, size=records).tolist(),
                                 'Sample': np.random.randint(5, size=records).tolist(),
                                 'AUC': np.random.randint(5, size=records).tolist()})

    print(tb)

    print(tb)

    df = tb.to_pandas()

    df_sum = df.groupby('Source').agg({target: 'count', 'Sample': 'nunique',
                                       'Drug1': 'nunique', 'Drug2': 'nunique'})

    print(df_sum)

    #
    # tb = tb.groupby(0, ['Source', 'Source'], [AggregationOp.COUNT, AggregationOp.MAX])
    #
    # print(tb)


def test_app_iloc():
    mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},

              {'a': 100, 'b': 200, 'c': 300, 'd': 400},

              {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]

    df = pd.DataFrame(mydict)

    print(df)
    print("iLOC")
    print("iloc method 1")
    print(df.iloc[:, 2])
    print("iloc method 2")
    print(df.iloc[2, :])

    print("LOC")
    print("loc method 1")
    print(df.loc[:, 'a'])
    print("loc method 2")
    print(df.loc[:, ['a', 'b']])
    print("loc method 3")


test_app_iloc()
