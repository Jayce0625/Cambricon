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
  DataType input1_dtype = DataType::FLOAT32;
  Dims input1_dims = Dims({-1, -1, -1, -1});
  ITensor *input1_tensor = network->AddInput(input1_dtype, input1_dims);
  CHECK_VALID(input1_tensor);

  DataType input2_dtype = DataType::FLOAT32;
  Dims input2_dims = Dims({-1, -1});
  ITensor *input2_tensor = network->AddInput(input2_dtype, input2_dims);
  CHECK_VALID(input2_tensor);

  // optional input
  std::vector<float> bias_value = {0.5};
  IConstNode *bias = network->AddIConstNode(DataType::FLOAT32, Dims({1}), bias_value.data());
  CHECK_VALID(bias);

  // create innerproduct node
  IInnerProductNode *Innerproduct = network->AddIInnerProductNode(input1_tensor,input2_tensor,bias->GetOutput(0));

  // mark network output
  for (auto i = 0; i < Innerproduct->GetOutputCount(); i++) {
    auto output_tensor = Innerproduct->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("innerproduct_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("innerproduct_model"));

  return 0;
}
