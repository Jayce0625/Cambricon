/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "mm_builder.h" // NOLINT 
#include "mm_network.h" // NOLINT
#include "mm_runtime.h" // NOLINT
#include "common/macros.h"
#include "common/data.h"
#include "common/container.h"
using namespace magicmind;  // NOLINT
int main(int argc, char **argv) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);

  // create input tensor
  DataType input1_dtype = DataType::FLOAT32;
  Dims input1_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input1_tensor = network->AddInput(input1_dtype, input1_dims);
  CHECK_VALID(input1_tensor);

  DataType input2_dtype = DataType::FLOAT32;
  Dims input2_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input2_tensor = network->AddInput(input2_dtype, input2_dims);
  CHECK_VALID(input2_tensor);

  // create minimum node
  IMinimumNode *Minimum = network->AddIMinimumNode(input1_tensor, input2_tensor);

  // mark network output
  for (auto i = 0; i < Minimum->GetOutputCount(); i++) {
    auto output_tensor = Minimum->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("minimum_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("minimum_model"));

  return 0;
}
