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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    LOG(ERROR) << "There should be two arguments with paths to csv files";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  arrow::BinaryBuilder left_id_builder(pool);
  arrow::BinaryBuilder right_id_builder(pool);
  arrow::BinaryBuilder cost_builder(pool);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("first", arrow::int64()), arrow::field("second", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  auto status = cylon::Table::FromArrow(ctx, argv[1], first_table);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  status = cylon::Table::FromCSV(ctx, argv[2], second_table, read_options);
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