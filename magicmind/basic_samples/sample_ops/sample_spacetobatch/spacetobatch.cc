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

  std::vector<int32_t> block_value = {2};
  magicmind::IConstNode *block_shape_tensor = network->AddIConstNode(DataType::INT32, Dims({1}), block_value.data());
  CHECK_VALID(block_shape_tensor);

  std::vector<int32_t> padding_value = {0, 0};
  magicmind::IConstNode *paddings_tensor = network->AddIConstNode(DataType::INT32, Dims({1, 2}), padding_value.data());
  CHECK_VALID(paddings_tensor);

  // create spacetobatch node
  ISpaceToBatchNode *Spacetobatch = network->AddISpaceToBatchNode(
      input_tensor, block_shape_tensor->GetOutput(0),
      paddings_tensor->GetOutput(0));
  // mark network output
  for (auto i = 0; i < Spacetobatch->GetOutputCount(); i++) {
    auto output_tensor = Spacetobatch->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(
      builder->BuildModel("spacetobatch_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("spacetobatch_model"));

  return 0;
}