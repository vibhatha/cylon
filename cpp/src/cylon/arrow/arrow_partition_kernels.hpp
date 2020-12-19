/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_ARROW_PARTITION_KERNELS_H
#define CYLON_ARROW_PARTITION_KERNELS_H

#include <memory>
#include <vector>
#include <arrow/api.h>
#include <glog/logging.h>
#include <cmath>
#include <util/macros.hpp>
#include <ctx/cylon_context.hpp>

#include "../data_types.hpp"
#include "../util/murmur3.hpp"
#include "../util/arrow_utils.hpp"
#include "../status.hpp"
#include "../arrow/arrow_types.hpp"
#include "../compute/aggregates.hpp"
#include "../net/mpi/mpi_operations.hpp"

namespace cylon {

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
static inline constexpr bool if_power2(T v) {
  return v && !(v & (v - 1));
}

template<typename ARROW_ARRAY_T>
using LoopRunner = std::function<void(const std::shared_ptr<ARROW_ARRAY_T> &casted_arr,
                                      uint64_t global_idx,
                                      uint64_t idx)>;

template<typename ARROW_ARRAY_T>
static inline void run_loop(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                            const LoopRunner<ARROW_ARRAY_T> &runner) {
  uint64_t offset = 0;
  for (const auto &arr: idx_col->chunks()) {
    const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
    for (int64_t i = 0; i < carr->length(); i++, offset++) {
      runner(carr, offset, i);
    }
  }
}

using Partitioner = std::function<uint32_t(uint64_t)>;

static inline Partitioner get_partitioner(uint32_t num_partitions) {
  if (if_power2(num_partitions)) {
    return [num_partitions](uint32_t val) {
      return val & (num_partitions - 1);
    };
  } else {
    return [num_partitions](uint32_t val) {
      return val % num_partitions;
    };
  }
}


class PartitionKernel {
 public:
  virtual ~PartitionKernel() = default;

  Status Partition(const std::shared_ptr<arrow::Array> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) {
    return Partition(std::make_shared<arrow::ChunkedArray>(idx_col),
                     num_partitions,
                     target_partitions,
                     partition_histogram);
  }

  virtual Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                           uint32_t num_partitions,
                           std::vector<uint32_t> &target_partitions,
                           std::vector<uint32_t> &partition_histogram) = 0;
};

class HashPartitionKernel : public PartitionKernel {
 public:
  Status UpdateHash(const std::shared_ptr<arrow::Array> &idx_col, std::vector<uint32_t> &partial_hashes) {
    return UpdateHash(std::make_shared<arrow::ChunkedArray>(idx_col), partial_hashes);
  }

  virtual Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                            std::vector<uint32_t> &partial_hashes) = 0;

  virtual inline uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) = 0;
};


template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_integer_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class ModuloPartitionKernel : public HashPartitionKernel {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_CTYPE = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &casted_arr,
                                                uint64_t chunk_offset,
                                                uint64_t idx) {
      uint32_t pseudo_hash = 31 * target_partitions[chunk_offset] + static_cast<uint32_t>(casted_arr->Value(idx));
      uint32_t p = partitioner(pseudo_hash);
      target_partitions[chunk_offset] = p;
      partition_histogram[p]++;
    };

    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }

    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &carr,
                                                uint64_t offset,
                                                uint64_t i) {
      partial_hashes[offset] = 31 * partial_hashes[offset] + static_cast<uint32_t>(carr->Value(i));
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  inline uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    const std::shared_ptr<ARROW_ARRAY_T> &carr = std::static_pointer_cast<ARROW_ARRAY_T>(values);
    return static_cast<uint32_t>(carr->Value(index));
  }
};


template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class NumericHashPartitionKernel : public HashPartitionKernel {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_CTYPE = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    const int len = sizeof(ARROW_CTYPE);
    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &carr,
                                                uint64_t global_idx,
                                                uint64_t idx) {
      ARROW_CTYPE val = carr->Value(idx);
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(&val, len, 0, &hash);
      hash += 31 * target_partitions[global_idx];
      uint32_t p = partitioner(hash);
      target_partitions[global_idx] = p;
      partition_histogram[p]++;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }
    const int len = sizeof(ARROW_CTYPE);
    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &carr,
                                                uint64_t global_idx,
                                                uint64_t idx) {
      ARROW_CTYPE val = carr->Value(idx);
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(&val, len, 0, &hash);
      hash += 31 * partial_hashes[global_idx];
      partial_hashes[global_idx] = hash;
    };
    run_loop(idx_col, loop_runner);
    return Status::OK();
  }

  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto reader = std::static_pointer_cast<ARROW_ARRAY_T>(values);
      ARROW_CTYPE lValue = reader->Value(index);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(&lValue, sizeof(ARROW_CTYPE), 0, &hash);
      return hash;
    }
  }
};

class FixedSizeBinaryHashPartitionKernel : public HashPartitionKernel {
 public:
  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto reader = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(reader->GetValue(index), reader->byte_width(), 0, &hash);
      return hash;
    }
  }

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    int32_t byte_width = std::static_pointer_cast<arrow::FixedSizeBinaryType>(idx_col->type())->byte_width();
    LoopRunner<arrow::FixedSizeBinaryArray> loop_runner = [&](const std::shared_ptr<arrow::FixedSizeBinaryArray> &carr,
                                                              uint64_t global_idx,
                                                              uint64_t idx) {
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(carr->GetValue(idx), byte_width, 0, &hash);
      hash += 31 * target_partitions[global_idx];
      uint32_t p = partitioner(hash);
      target_partitions[global_idx] = p;
      partition_histogram[p]++;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }

    int32_t byte_width = std::static_pointer_cast<arrow::FixedSizeBinaryType>(idx_col->type())->byte_width();
    LoopRunner<arrow::FixedSizeBinaryArray> loop_runner = [&](const std::shared_ptr<arrow::FixedSizeBinaryArray> &carr,
                                                              uint64_t global_idx,
                                                              uint64_t idx) {
      uint32_t hash = 0;
      util::MurmurHash3_x86_32(carr->GetValue(idx), byte_width, 0, &hash);
      hash += 31 * partial_hashes[global_idx];
      partial_hashes[global_idx] = hash;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }
};

class BinaryHashPartitionKernel : public HashPartitionKernel {
 public:
  uint32_t ToHash(const std::shared_ptr<arrow::Array> &values, int64_t index) override {
    if (values->IsNull(index)) {
      return 0;
    } else {
      auto reader = std::static_pointer_cast<arrow::BinaryArray>(values);
      int length = 0;
      const uint8_t *val = reader->GetValue(index, &length);
      uint32_t hash = 0;
      cylon::util::MurmurHash3_x86_32(val, length, 0, &hash);
      return hash;
    }
  }

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    if (partition_histogram.size() != num_partitions
        || target_partitions.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "target partitions or histogram not initialized!");
    }

    Partitioner partitioner = get_partitioner(num_partitions);
    LoopRunner<arrow::BinaryArray> loop_runner = [&](const std::shared_ptr<arrow::BinaryArray> &carr,
                                                     uint64_t global_idx,
                                                     uint64_t idx) {
      int32_t byte_width = 0;
      uint32_t hash = 0;
      const uint8_t *val = carr->GetValue(idx, &byte_width);
      util::MurmurHash3_x86_32(val, byte_width, 0, &hash);
      hash += 31 * target_partitions[global_idx];
      uint32_t p = partitioner(hash);
      target_partitions[global_idx] = p;
      partition_histogram[p]++;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

  Status UpdateHash(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                    std::vector<uint32_t> &partial_hashes) override {
    if (partial_hashes.size() != (size_t) idx_col->length()) {
      return Status(Code::Invalid, "partial hashes size != idx col length!");
    }

    LoopRunner<arrow::BinaryArray> loop_runner = [&](const std::shared_ptr<arrow::BinaryArray> &carr,
                                                     uint64_t global_idx,
                                                     uint64_t idx) {
      int32_t byte_width = 0;
      uint32_t hash = 0;
      const uint8_t *val = carr->GetValue(idx, &byte_width);
      util::MurmurHash3_x86_32(val, byte_width, 0, &hash);
      hash += 31 * partial_hashes[global_idx];
      partial_hashes[global_idx] = hash;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }
};

using UInt8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt8Type>;
using UInt16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt16Type>;
using UInt32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt32Type>;
using UInt64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt64Type>;
using Int8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int8Type>;
using Int16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int16Type>;
using Int32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int32Type>;
using Int64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int64Type>;
using HalfFloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::HalfFloatType>;
using FloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::FloatType>;
using DoubleArrayHashPartitioner = NumericHashPartitionKernel<arrow::DoubleType>;
using StringHashPartitioner = BinaryHashPartitionKernel;
using BinaryHashPartitioner = BinaryHashPartitionKernel;

/**
 * creates hash partition kernel
 * @param data_type
 * @return
 */
std::unique_ptr<HashPartitionKernel> CreateHashPartitionKernel(const std::shared_ptr<arrow::DataType> &data_type);


class RowHashingKernel {
 private:
  std::vector<std::unique_ptr<HashPartitionKernel>> hash_kernels;
 public:
  explicit RowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields);

  int32_t Hash(const std::shared_ptr<arrow::Table> &table, int64_t row);
};

class PartialRowHashingKernel {
 private:
  std::vector<std::unique_ptr<HashPartitionKernel>> hash_kernels;
  std::vector<int> columns;
 public:
  explicit PartialRowHashingKernel(const std::vector<std::shared_ptr<arrow::Field>> &fields, const std::vector<int> &cols);

  int32_t Hash(const std::shared_ptr<arrow::Table> &table, int64_t row);
};


template<typename ARROW_T, typename = typename std::enable_if<
    arrow::is_number_type<ARROW_T>::value | arrow::is_boolean_type<ARROW_T>::value>::type>
class RangePartitionKernel : public PartitionKernel {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using ARROW_SCALAR_T = typename arrow::TypeTraits<ARROW_T>::ScalarType;
  using ARROW_C_T = typename arrow::TypeTraits<ARROW_T>::CType;

 public:
  RangePartitionKernel(std::shared_ptr<CylonContext> &ctx,
                       bool ascending,
                       uint64_t num_samples,
                       uint32_t num_bins) : PartitionKernel(),
                                            ascending(ascending),
                                            num_bins(num_bins),
                                            num_samples(num_samples),
                                            ctx(ctx) {
  };

  Status Partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col,
                   uint32_t num_partitions,
                   std::vector<uint32_t> &target_partitions,
                   std::vector<uint32_t> &partition_histogram) override {
    auto status = build_bin_to_partition(idx_col, num_partitions);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    // resize vectors
    partition_histogram.resize(num_partitions, 0);
    target_partitions.resize(idx_col->length());

    LoopRunner<ARROW_ARRAY_T> loop_runner = [&](const std::shared_ptr<ARROW_ARRAY_T> &carr,
                                                uint64_t global_idx,
                                                uint64_t idx) {
      ARROW_C_T val = carr->Value(idx);
      uint32_t p = ascending ?
                   bin_to_partition[get_bin_pos(val)] : num_partitions - 1 - bin_to_partition[get_bin_pos(val)];
      target_partitions[global_idx] = p;
      partition_histogram[p]++;
    };
    run_loop(idx_col, loop_runner);

    return Status::OK();
  }

 private:
  Status inline build_bin_to_partition(const std::shared_ptr<arrow::ChunkedArray> &idx_col, uint32_t num_partitions) {
    const std::shared_ptr<DataType> &data_type = tarrow::ToCylonType(idx_col->type());
    std::shared_ptr<arrow::ChunkedArray> sampled_array;

    if ((uint64_t) idx_col->length() == num_samples) { // if len == num_samples, dont sample, just use idx col as it is!
      sampled_array = std::make_shared<arrow::ChunkedArray>(idx_col->chunks());
    } else { // else, sample idx_col for num_samples
      std::shared_ptr<arrow::Array> samples;
      auto a_status = util::SampleArray(idx_col, num_samples, samples);
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(a_status)
      sampled_array = std::make_shared<arrow::ChunkedArray>(samples);
    }

    // calculate minmax of the sample
    std::shared_ptr<compute::Result> minmax;
    Status status = compute::MinMax(ctx, sampled_array, data_type, minmax);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    const auto &struct_scalar = minmax->GetResult().scalar_as<arrow::StructScalar>();
    min = std::static_pointer_cast<ARROW_SCALAR_T>(struct_scalar.value[0])->value;
    max = std::static_pointer_cast<ARROW_SCALAR_T>(struct_scalar.value[1])->value;
    if (arrow::is_integer_type<ARROW_T>()) {
      range_per_bin = (max - min + num_bins - 1) / num_bins; // upper bound of the division
    } else {
      range_per_bin = (max - min) / num_bins; // upper bound of the division
    }
    max = min + range_per_bin * num_bins; // update max

    // create sample histogram
    std::vector<uint64_t> local_counts(num_bins + 2, 0);
    for (const auto &arr: sampled_array->chunks()) {
      const std::shared_ptr<ARROW_ARRAY_T> &casted_arr = std::static_pointer_cast<ARROW_ARRAY_T>(arr);
      for (int64_t i = 0; i < casted_arr->length(); i++) {
        ARROW_C_T val = casted_arr->Value(i);
        local_counts[get_bin_pos(val)]++;
      }
    }

    // all reduce local sample histograms
    std::vector<uint64_t> global_counts, *global_counts_ptr;
    if (ctx->GetWorldSize() > 1) { // if distributed, all-reduce all local bin counts
      global_counts.resize(num_bins + 2, 0);
      status = cylon::mpi::AllReduce(local_counts.data(), global_counts.data(), num_bins + 2, cylon::UInt64(),
                                     cylon::net::SUM);
      RETURN_CYLON_STATUS_IF_FAILED(status)
      global_counts_ptr = &global_counts;
      local_counts.clear();
    } else { // else, just use local bin counts
      global_counts_ptr = &local_counts;
    }

    float_t quantile = 1.0 / num_partitions, prefix_sum = 0;

    LOG(INFO) << "len=" << idx_col->length() << " min=" << min << " max=" << max << " range per bin=" <<
              range_per_bin << " num bins=" << num_bins << " quantile=" << quantile;

    // divide global histogram into quantiles
    const uint64_t total_samples = ctx->GetWorldSize() * num_samples;
    uint32_t curr_partition = 0;
    float_t target_quantile = quantile;
    for (const auto &c:*global_counts_ptr) {
      bin_to_partition.push_back(curr_partition);
      float_t freq = (float_t) c / total_samples;
      prefix_sum += freq;
      if (prefix_sum > target_quantile) {
        curr_partition += (curr_partition < num_partitions - 1); // if curr_partition < numpartition: curr_partition++
        target_quantile += quantile;
      }
    }

    if (curr_partition != num_partitions - 1) {
      LOG(WARNING) << "sample was unable to capture distribution. try increasing num_samples or num_bins";
    }
    return Status::OK();
  }

  inline constexpr size_t get_bin_pos(ARROW_C_T val) {
    return (val >= min)
        + ((val >= min) & (val < max)) * ((val - min) / range_per_bin)
        + (val >= max) * num_bins;
  }

  const bool ascending;
  const uint32_t num_bins;
  const uint64_t num_samples;
  std::shared_ptr<CylonContext> ctx; // todo dont need a copy here
  std::vector<uint32_t> bin_to_partition;
  ARROW_C_T min, max, range_per_bin;
};

/**
 * create range partition
 * @param data_type
 * @param num_partitions
 * @param ctx
 * @param ascending
 * @param num_samples
 * @param num_bins
 * @return
 */
std::unique_ptr<PartitionKernel> CreateRangePartitionKernel(const std::shared_ptr<arrow::DataType> &data_type,
                                                            std::shared_ptr<CylonContext> &ctx,
                                                            bool ascending,
                                                            uint64_t num_samples,
                                                            uint32_t num_bins);

}  // namespace cylon

#endif //CYLON_ARROW_PARTITION_KERNELS_H
