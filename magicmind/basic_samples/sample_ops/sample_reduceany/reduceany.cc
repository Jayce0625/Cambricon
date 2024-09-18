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
  DataType input_dtype = DataType::BOOL;
  Dims input_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  // optional input
  std::vector<int32_t> axis_value = {1};
  magicmind::IConstNode *axis = network->AddIConstNode(DataType::INT32, Dims({1}), axis_value.data());
  CHECK_VALID(axis);

  // create reduce_any node
  IReduceNode *ReduceAny = network->AddIReduceNode(input_tensor,axis->GetOutput(0),IReduce::OR,false);

  //using ReduceAny default paramters, you can set each attribute's value.
  //CHECK_STATUS(ReduceAny->SetKeepDims([value]));
  //CHECK_STATUS(ReduceAny->SetAlpha([value]));
  //CHECK_STATUS(ReduceAny->SetBeta([value]));
  //CHECK_STATUS(ReduceAny->SetShapeinferZero([value]));

  // mark network output
  for (auto i = 0; i < ReduceAny->GetOutputCount(); i++) {
    auto output_tensor = ReduceAny->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("reduce_any_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("reduce_any_model"));

  return 0;
}
