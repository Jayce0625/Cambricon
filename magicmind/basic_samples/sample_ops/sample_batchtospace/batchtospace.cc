
/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "common/container.h"
#include "common/data.h"
#include "common/macros.h"
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
using namespace magicmind;  // NOLINT
int main(int argc, char **argv) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);

  // create input tensor
  DataType input_dtype = DataType::FLOAT32;
  Dims input_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  DataType block_shape_dtype = DataType::INT32;  
  Dims block_shape_dims = Dims({1});
  std::vector<int32_t> block_shape_value = {2};
  magicmind::ITensor *block_shape_tensor = network->AddIConstNode(block_shape_dtype, block_shape_dims,block_shape_value.data())->GetOutput(0);
  CHECK_VALID(block_shape_tensor);

  DataType crops_dtype = DataType::INT32;
  Dims crops_dims = Dims({1, 2});
  std::vector<int32_t> crops_value = {0, 0};
  magicmind::ITensor *crops_tensor = network->AddIConstNode(crops_dtype, crops_dims, crops_value.data())->GetOutput(0);
  CHECK_VALID(crops_tensor);

  // create batch2space node
  IBatchToSpaceNode *Batch2space = network->AddIBatchToSpaceNode(input_tensor,block_shape_tensor,crops_tensor);

  // mark network output
  for (auto i = 0; i < Batch2space->GetOutputCount(); i++) {
    auto output_tensor = Batch2space->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(
      builder->BuildModel("batchtospace_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("batchtospace_model"));

  return 0;
}