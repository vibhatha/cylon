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
#include <arrow/array.h>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <math.h>

template< typename T > std::array< uint8_t, sizeof(T) >  to_bytes( const T& object ){
  std::array< uint8_t, sizeof(T) > bytes ;

  const uint8_t* begin = reinterpret_cast< const uint8_t* >( std::addressof(object) ) ;
  const uint8_t* end = begin + sizeof(T) ;
  std::copy( begin, end, std::begin(bytes) ) ;

  return bytes ;
}

unsigned long long Randomize() {
  unsigned long long randnumber = 0;
  int digits[20];

  for (int i = 19; i >= 1; i--) {
    digits[i]=rand()%10;
  }
  for(int i=19; i>=1; i--) {
    unsigned long long power = pow(10, i-1);
    if (power%2 != 0 && power != 1) {
      power++;
    }
    randnumber += power * digits[i];
  }
  return randnumber;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be one argument with count";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  arrow::FixedSizeBinaryBuilder left_id_builder(arrow::fixed_size_binary(8), pool);
  arrow::FixedSizeBinaryBuilder right_id_builder(arrow::fixed_size_binary(8), pool);
  arrow::FixedSizeBinaryBuilder cost_builder(arrow::fixed_size_binary(8), pool);

  uint64_t count = std::stoull(argv[1]);
  /* Seed */
  std::random_device rd;

  /* Random number generator */
  std::default_random_engine generator(rd() + ctx->GetRank());

  /* Distribution on which to apply the generator */
  uint64_t range = count * ctx->GetWorldSize();
  std::uniform_int_distribution<uint64_t> distribution(0, range);
  uint8_t lb[8];
  uint8_t rb[8];
  uint8_t vb[8];
  uint64_t max = 0;
  srand(time(NULL) + ctx->GetRank());

  for (int i = 0; i < count; i++) {
    uint64_t l = Randomize() % range;
    uint64_t r = Randomize() % range;
    uint64_t v = Randomize() % range;

    if (l > max) {
      max = l;
    }
    for (int i = 0; i < 8; i++) {
      lb[i] = (l >> (i * 8)) & 0XFF;
      rb[i] = (r >> (i * 8)) & 0XFF;
      vb[i] = (v >> (i * 8)) & 0XFF;
    }

    arrow::Status st = left_id_builder.Append(lb);
    st = right_id_builder.Append(rb);
    st = cost_builder.Append(vb);
  }

  std::cout << "****** MAX *************** " << max << " range " << range << " " << count << "X" << ctx->GetWorldSize() << std::endl;

  std::shared_ptr<arrow::Array> left_id_array;
  arrow::Status st = left_id_builder.Finish(&left_id_array);
  std::shared_ptr<arrow::Array> right_id_array;
  st = right_id_builder.Finish(&right_id_array);

  std::shared_ptr<arrow::Array> cost_array;
  st = cost_builder.Finish(&cost_array);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("first", arrow::fixed_size_binary(8)), arrow::field("second", arrow::fixed_size_binary(8))};
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

  status = cylon::Table::DistributedJoin(first_table, second_table,
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