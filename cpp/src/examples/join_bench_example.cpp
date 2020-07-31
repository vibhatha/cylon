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

#include <glog/logging.h>
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <chrono>
#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array.h>
#include <random>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be one argument with count";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  arrow::BinaryBuilder left_id_builder(pool);
  arrow::BinaryBuilder right_id_builder(pool);
  arrow::BinaryBuilder cost_builder(pool);

  int count = std::stoi(argv[1]);
  /* Seed */
  std::random_device rd;

  /* Random number generator */
  std::default_random_engine generator(rd());

  /* Distribution on which to apply the generator */
  uint64_t range = count * ctx->GetWorldSize();
  std::uniform_int_distribution<uint64_t> distribution(0, range);
  uint8_t lb[8];
  uint8_t rb[8];
  uint8_t vb[8];
  for (int i = 0; i < count; i++) {
    uint64_t l = distribution(generator);
    uint64_t r = distribution(generator);
    uint64_t v = distribution(generator);

    for (int i = 0; i < 8; i++) {
      lb[7 - i] = (l >> (i * 8));
      rb[7 - i] = (r >> (i * 8));
      vb[7 - i] = (v >> (i * 8));
    }

    arrow::Status st = left_id_builder.Append(lb, 8);
    st = right_id_builder.Append(rb, 8);
    st = cost_builder.Append(vb, 8);
  }

  std::shared_ptr<arrow::Array> left_id_array;
  arrow::Status st = left_id_builder.Finish(&left_id_array);
  std::shared_ptr<arrow::Array> right_id_array;
  st = right_id_builder.Finish(&right_id_array);

  std::shared_ptr<arrow::Array> cost_array;
  st = cost_builder.Finish(&cost_array);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("first", arrow::binary()), arrow::field("second", arrow::binary())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> left_table = arrow::Table::Make(schema, {left_id_array, cost_array});
  std::shared_ptr<arrow::Table> right_table = arrow::Table::Make(schema, {right_id_array, cost_array});

  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  auto status = cylon::Table::FromArrowTable(ctx, left_table, &first_table);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  status = cylon::Table::FromArrowTable(ctx, right_table, &second_table);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[2];
    ctx->Finalize();
    return 1;
  }
  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(read_end_time - start_start).count() << "[ms]";

  status = first_table->DistributedJoin(second_table,
                                        cylon::join::config::JoinConfig::InnerJoin(0, 0), &joined);
  if (!status.is_ok()) {
    LOG(INFO) << "Table join failed ";
    ctx->Finalize();
    return 1;
  }
  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
            << second_table->Rows() << ", Joined has : " << joined->Rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(join_end_time - read_end_time).count() << "[ms]";
  ctx->Finalize();
  return 0;
}