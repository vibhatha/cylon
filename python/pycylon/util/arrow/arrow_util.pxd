##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
from pyarrow.lib cimport CTable as CArrowTable
from libcpp.memory cimport shared_ptr
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.table cimport CTable
from pycylon.data.table import Table
from pyarrow.includes.libarrow cimport CStatus as ArrowCStatus
from pyarrow.includes.libarrow cimport CMemoryPool as ArrowCMemoryPool
from pyarrow.lib cimport maybe_unbox_memory_pool


cdef extern from "../../../cpp/src/cylon/ctx/arrow_memory_utils.hpp" namespace "cylon":
    ArrowCMemoryPool *ToArrowPool(const shared_ptr[CCylonContext] &ctx);


cdef extern from "../../../cpp/src/cylon/util/arrow_utils.hpp" namespace "cylon::util":
    @staticmethod
    ArrowCStatus Duplicate(const shared_ptr[CArrowTable] & table, ArrowCMemoryPool*pool, shared_ptr[CArrowTable] & out);
