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
using namespace magicmind;  // NOLINT
int main(int argc, char **argv) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);

  // create input tensor
  DataType input_dtype = DataType::FLOAT32;
  Dims input_dims = Dims({1,4,8,8});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  std::vector<int32_t> shape = {1,16,16,1};
  magicmind::IConstNode *shape_tensor = network->AddIConstNode(DataType::INT32, Dims({4}), shape.data());
  CHECK_VALID(shape_tensor);

  // create reshape node
  IReshapeNode *Reshape = network->AddIReshapeNode(input_tensor,shape_tensor->GetOutput(0));

  //using Reshape default paramters, you can set each attribute's value.
  // CHECK_STATUS(Reshape->SetAxis([0]));
  //CHECK_STATUS(Reshape->SetNumAxes([value]));
  //CHECK_STATUS(Reshape->SetAllowZero([value]));

  // mark network output
  for (auto i = 0; i < Reshape->GetOutputCount(); i++) {
    auto output_tensor = Reshape->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("reshape_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("reshape_model"));

  return 0;
}