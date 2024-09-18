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
  Dims input_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  auto filter_dim = magicmind::Dims({1, 2, 3, 4});
  std::vector<float> filter_buffer = GenRand(filter_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *weight = network->AddIConstNode(DataType::FLOAT32, filter_dim, filter_buffer.data());
  CHECK_VALID(weight);

  // optional input
  std::vector<int32_t> bias_value = {1};
  magicmind::IConstNode *bias = network->AddIConstNode(DataType::FLOAT32, Dims({1}), bias_value.data());
  CHECK_VALID(bias);

  // create conv2d node
  IConvNode *Conv2d = network->AddIConvNode(input_tensor,weight->GetOutput(0),bias->GetOutput(0));

  //using Conv2d default paramters, you can set each attribute's value.
  //CHECK_STATUS(Conv2d->SetPaddingMode([value]));
  //CHECK_STATUS(Conv2d->SetPad([value]));
  //CHECK_STATUS(Conv2d->SetStride([value]));
  //CHECK_STATUS(Conv2d->SetDilation([value]));
  //CHECK_STATUS(Conv2d->SetLayout([value]));
  //CHECK_STATUS(Conv2d->SetGroup([value]));

  // mark network output
  for (auto i = 0; i < Conv2d->GetOutputCount(); i++) {
    auto output_tensor = Conv2d->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("conv2d_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("conv2d_model"));

  return 0;
}
