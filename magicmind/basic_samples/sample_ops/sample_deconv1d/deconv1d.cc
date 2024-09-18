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
  Dims input_dims = Dims({10, 24, 4});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  auto filter_dim = magicmind::Dims({4, 4, 4});
  std::vector<float> filter_buffer = GenRand(filter_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *weight = network->AddIConstNode(DataType::FLOAT32, filter_dim, filter_buffer.data());
  CHECK_VALID(weight);

  // optional input
  // std::vector<float> bias_value = {0, 0, 0, 0};
  // magicmind::IConstNode *bias = network->AddIConstNode(DataType::FLOAT32, Dims({4}), bias_value.data());
  // CHECK_VALID(bias);

  // optional input
  // DataType output_shape_dtype = DataType::INT32;
  // Dims output_shape_dims = Dims({-1});
  // magicmind::ITensor *output_shape_tensor = network->AddInput(output_shape_dtype, output_shape_dims);
  // CHECK_VALID(output_shape_tensor);

  // create deconv1d node
  IDeconv1DNode *Deconv1d = network->AddIDeconv1DNode(input_tensor,weight->GetOutput(0), /*bias->GetOutput(0)*/nullptr, nullptr);

  //using Deconv1d default paramters, you can set each attribute's value.
  CHECK_STATUS(Deconv1d->SetPaddingMode(IPaddingMode::EXPLICIT));
  CHECK_STATUS(Deconv1d->SetPad(0, 0));
  CHECK_STATUS(Deconv1d->SetStride(1));
  CHECK_STATUS(Deconv1d->SetDilation(1));
  CHECK_STATUS(Deconv1d->SetGroup(1));
  CHECK_STATUS(Deconv1d->SetLayout(Layout::NTC, Layout::NTC, Layout::NTC));
  // CHECK_STATUS(Deconv1d->SetOutpad(value));

  // mark network output
  for (auto i = 0; i < Deconv1d->GetOutputCount(); i++) {
    auto output_tensor = Deconv1d->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("deconv1d_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("deconv1d_model"));

  return 0;
}
