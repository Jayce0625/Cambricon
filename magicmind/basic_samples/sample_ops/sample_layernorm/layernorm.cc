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


  const int64_t c = 3, h = 16, w = 16;
//   const int64_t numel = c * h * w;

  // create input tensor
  DataType input1_dtype = DataType::FLOAT32;
  Dims input1_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input1_dtype, input1_dims);
  CHECK_VALID(input_tensor);

  // weight
//   auto filter_dim = magicmind::Dims({1, 2, 3, 4});
  auto filter_dim = magicmind::Dims({c, h, w});
  std::vector<float> filter_buffer = GenRand(filter_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *weight = network->AddIConstNode(DataType::FLOAT32, filter_dim, filter_buffer.data());
  CHECK_VALID(weight);

  // bias
  auto bias_dim = magicmind::Dims({c, h, w});
  std::vector<float> bias_buffer = GenRand(bias_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *bias = network->AddIConstNode(DataType::FLOAT32, bias_dim, bias_buffer.data());
  CHECK_VALID(bias);
  
  ILayerNormNode *layernorm = network->AddILayerNormNode(
        input_tensor, weight->GetOutput(0), bias->GetOutput(0), {c, h, w}, 0.1);

  // mark network output
  for (auto i = 0; i < layernorm->GetOutputCount(); i++) {
    auto output_tensor = layernorm->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("layernorm_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("layernorm_model"));

  return 0;
}