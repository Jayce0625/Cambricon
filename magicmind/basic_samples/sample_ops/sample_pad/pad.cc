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

  // create paddings as a const node
  int32_t paddings_value[] = {1, 1, 1, 1};
  auto paddings_node = network->AddIConstNode(magicmind::DataType::INT32,
      magicmind::Dims({4}), paddings_value);
  auto paddings = paddings_node->GetOutput(0);

  // create pad node
  auto pad_node = network->AddIPadNode(input, paddings, nullptr);

  // mark network output
  CHECK_STATUS(network->MarkOutput(pad_node->GetOutput(0)));

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("pad_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("pad_model"));

  return 0;
}
