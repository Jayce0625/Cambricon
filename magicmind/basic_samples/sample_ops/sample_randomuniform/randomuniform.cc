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

  // create shape tensor
  std::vector<int32_t> shape_value = {3, 2, 4, 5};
  magicmind::IConstNode *shape = network->AddIConstNode(DataType::INT32, Dims({4}), shape_value.data());
  magicmind::ITensor *shape_tensor = shape->GetOutput(0);
  CHECK_VALID(shape_tensor);

  // create high tensor
  std::vector<float> high_value = {10.0};
  magicmind::IConstNode *high = network->AddIConstNode(DataType::FLOAT32, Dims({1}), high_value.data());
  magicmind::ITensor *high_tensor = high->GetOutput(0);
  CHECK_VALID(high_tensor);

  // create low tensor
  std::vector<float> low_value = {1.0};
  magicmind::IConstNode *low = network->AddIConstNode(DataType::FLOAT32, Dims({1}), low_value.data());
  magicmind::ITensor *low_tensor = low->GetOutput(0);
  CHECK_VALID(low_tensor);

  // create random_uniform node
  IRandomUniformNode *RandomUniform = network->AddIRandomUniformNode(
                                            shape_tensor, high_tensor, low_tensor, DataType::FLOAT32);

  //using RandomUniform default paramters, you can set each attribute's value.
  //CHECK_STATUS(RandomUniform->SetSeed([value]));

  // mark network output
  for (auto i = 0; i < RandomUniform->GetOutputCount(); i++) {
    auto output_tensor = RandomUniform->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("randomuniform_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("randomuniform_model"));

  return 0;
}
