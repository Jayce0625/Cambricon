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

  // create input, alpha, beta
  DataType dtype = DataType::FLOAT32;
  Dims input_dims = Dims({2, 3, 2});
  magicmind::ITensor *input_tensor = network->AddInput(dtype, input_dims);
  CHECK_VALID(input_tensor);

  Dims alpha_dims = Dims({1, 1, 2});
  magicmind::ITensor *alpha_tensor = network->AddInput(dtype, alpha_dims);
  CHECK_VALID(alpha_tensor);

  Dims beta_dims = Dims({1, 1, 2});
  magicmind::ITensor *beta_tensor = network->AddInput(dtype, beta_dims);
  CHECK_VALID(beta_tensor);

  // create scale node
  IScaleNode *Scale = network->AddIScaleNode(input_tensor, alpha_tensor, beta_tensor);

  // Axis
  CHECK_STATUS(Scale->SetAxis(0));

  // mark network output
  for (auto i = 0; i < Scale->GetOutputCount(); i++) {
    auto output_tensor = Scale->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("scale_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("scale_model"));

  return 0;
}