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
  DataType values_dtype = DataType::FLOAT32;
  Dims values_dims = Dims({-1, -1});
  magicmind::ITensor *values_tensor = network->AddInput(values_dtype, values_dims);
  CHECK_VALID(values_tensor);

  DataType indices_dtype = DataType::INT32;
  Dims indices_dims = Dims({2, 2});
  magicmind::ITensor *indices_tensor = network->AddInput(indices_dtype, indices_dims);
  CHECK_VALID(indices_tensor);

  std::vector<int32_t> axis_value = {1};
  magicmind::IConstNode *axis = network->AddIConstNode(DataType::INT32, Dims({1}), axis_value.data());
  CHECK_VALID(axis);

  // create gather_elw node
  IGatherElwNode *GatherElw = network->AddIGatherElwNode(values_tensor,indices_tensor,axis->GetOutput(0));

  // mark network output
  for (auto i = 0; i < GatherElw->GetOutputCount(); i++) {
    auto output_tensor = GatherElw->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("gather_elw_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("gather_elw_model"));

  return 0;
}
