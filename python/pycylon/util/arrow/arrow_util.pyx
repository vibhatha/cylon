from pyarrow.includes.libarrow cimport CMemoryPool as ArrowCMemoryPool
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.table cimport CTable
from pycylon.data.table import Table
from pyarrow.includes.libarrow cimport CStatus as ArrowCStatus
from pyarrow.lib cimport maybe_unbox_memory_pool
from pycylon.api.lib cimport (pycylon_wrap_context,
pycylon_unwrap_context,
pycylon_unwrap_table,
pycylon_wrap_table)


cdef ArrowCMemoryPool* to_arrow_pool(ctx):
    cdef shared_ptr[CCylonContext] c_ctx = pycylon_unwrap_context(ctx)
    return ToArrowPool(c_ctx)
