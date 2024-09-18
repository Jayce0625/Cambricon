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
  ITensor *input_tensor = network->AddInput(DataType::FLOAT32, Dims({-1, -1, -1, -1}));
  CHECK_VALID(input_tensor);

  // create src tensor
  ITensor *src_tensor = network->AddInput(DataType::FLOAT32, Dims({-1, -1, -1, -1}));
  CHECK_VALID(src_tensor);

  // create index
  std::vector<int32_t> index_value = {1,1,1,1};
  IConstNode *index = network->AddIConstNode(DataType::INT32, Dims({1,1,1,4}), index_value.data());
  CHECK_VALID(index);

  // create axis
  std::vector<int32_t> axis_value = {1};
  IConstNode *axis = network->AddIConstNode(DataType::INT32, Dims({1}), axis_value.data());
  CHECK_VALID(axis);

  // create scatter node
  IScatterNode *Scatter = network->AddIScatterNode(input_tensor, axis->GetOutput(0),
		                                index->GetOutput(0), src_tensor, IScatter::SCATTER);

  // mark network output
  for (auto i = 0; i < Scatter->GetOutputCount(); i++) {
    auto output_tensor = Scatter->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("scatter_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("scatter_model"));

  return 0;
}
