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

  // create reduce_norm1 node
  IReduceNode *ReduceNorm1 = network->AddIReduceNode(input_tensor,axis->GetOutput(0),IReduce::NORM1,false);

  //using ReduceNorm1 default paramters, you can set each attribute's value.
  //RCHECK_STATUS(educeNorm1->SetKeepDims([value]));
  //CHECK_STATUS(ReduceNorm1->SetHasNumAxes([value]));
  //CHECK_STATUS(ReduceNorm1->SetAlpha([value]));
  //CHECK_STATUS(ReduceNorm1->SetBeta([value]));
  //CHECK_STATUS(ReduceNorm1->SetShapeinferZero([value]));

  // mark network output
  for (auto i = 0; i < ReduceNorm1->GetOutputCount(); i++) {
    auto output_tensor = ReduceNorm1->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("reduce_norm1_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("reduce_norm1_model"));

  return 0;
}
