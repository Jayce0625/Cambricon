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

  // create min tensor
  DataType min_dtype = DataType::FLOAT32;
  Dims min_dims = Dims({1});
  std::vector<float> min_value = {2};
  magicmind::IConstNode *min = network->AddIConstNode(min_dtype, min_dims,
                                                       min_value.data());
  CHECK_VALID(min);
  
  // create max tensor
  DataType max_dtype = DataType::FLOAT32;
  Dims max_dims = Dims({1});
  std::vector<float> max_value = {3};
  magicmind::IConstNode *max = network->AddIConstNode(max_dtype, max_dims,
                                                       max_value.data());
  CHECK_VALID(max);
  
  // create clip node
  IClipNode *Clip = network->AddIClipNode(input_tensor, min->GetOutput(0), max->GetOutput(0));

  // mark network output
  for (auto i = 0; i < Clip->GetOutputCount(); i++) {
    auto output_tensor = Clip->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("clip_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("clip_model"));

  return 0;
}
