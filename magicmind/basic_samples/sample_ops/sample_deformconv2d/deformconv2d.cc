/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "mm_builder.h"  // NOLINT 
#include "mm_network.h"  // NOLINT
#include "mm_runtime.h"  // NOLINT
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

  DataType offset_dtype = DataType::FLOAT32;
  Dims offset_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *offset_tensor = network->AddInput(offset_dtype, offset_dims);
  CHECK_VALID(offset_tensor);

  // optional input
  DataType mask_dtype = DataType::FLOAT32;
  Dims mask_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *mask_tensor = network->AddInput(mask_dtype, mask_dims);
  CHECK_VALID(mask_tensor);

  // optional input
  std::vector<float> bias_value = {1};
  magicmind::IConstNode *bias =
    network->AddIConstNode(DataType::FLOAT32, Dims({1}), bias_value.data());
  CHECK_VALID(bias);

  // create deformconv2d node
  IDeformConv2DNode *Deformconv2d = network->AddIDeformConv2DNode(
      input_tensor, weight->GetOutput(0), offset_tensor, mask_tensor, bias->GetOutput(0));

  // mark network output
  for (auto i = 0; i < Deformconv2d->GetOutputCount(); i++) {
    auto output_tensor = Deformconv2d->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(
      builder->BuildModel("deformconv2d_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("deformconv2d_model"));

  return 0;
}
