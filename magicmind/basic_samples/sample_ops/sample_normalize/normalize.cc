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

  // optional input
  std::vector<int32_t> axis_value = {1};
  magicmind::IConstNode *axis = network->AddIConstNode(DataType::INT32, Dims({1}), axis_value.data());
  CHECK_VALID(axis);

  // create normalize node
  INormalizeNode *Normalize = network->AddINormalizeNode(input_tensor,axis->GetOutput(0), nullptr);

  //using Normalize default paramters, you can set each attribute's value.
  //CHECK_STATUS(Normalize->SetP([value]));
  //CHECK_STATUS(Normalize->SetEpsilon([value]));
  //CHECK_STATUS(Normalize->SetAcrossSpatial([value]));
  //CHECK_STATUS(Normalize->SetChannelShared([value]));

  // mark network output
  for (auto i = 0; i < Normalize->GetOutputCount(); i++) {
    auto output_tensor = Normalize->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("normalize_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("normalize_model"));

  return 0;
}