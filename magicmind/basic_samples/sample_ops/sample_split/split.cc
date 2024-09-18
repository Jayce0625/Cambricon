/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
#include "common/macros.h"
#include "common/data.h"
#include "common/container.h"

int main(int argc, char **argv) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);

  // create input tensor
  auto dims = magicmind::Dims({-1, -1, -1, -1});
  auto input = network->AddInput(magicmind::DataType::FLOAT32, dims);
  CHECK_VALID(input);

  // create axis as a const node
  int32_t axis_value[] = {2};
  auto axis_node = network->AddIConstNode(magicmind::DataType::INT32,
      magicmind::Dims(std::vector<int64_t>{}), axis_value);
  auto axis = axis_node->GetOutput(0);

  // create split node
  int64_t split_num = 2;  // the input tensor will be split into [split_num] tensors.
  auto split_node = network->AddISplitNode(input, axis, split_num);

  // mark network output
  for (int i = 0; i < split_node->GetOutputCount(); ++i) {
    CHECK_STATUS(network->MarkOutput(split_node->GetOutput(i)));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("split_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("split_model"));

  return 0;
}
