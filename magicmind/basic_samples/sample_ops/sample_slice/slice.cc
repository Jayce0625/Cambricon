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
  DataType input1_dtype = DataType::FLOAT32;
  Dims input1_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input1_tensor = network->AddInput(input1_dtype, input1_dims);
  CHECK_VALID(input1_tensor);

  // create begin tensor
  std::vector<int32_t> begin_value = {0};
  magicmind::IConstNode *begin = network->AddIConstNode(DataType::INT32, Dims({1}), begin_value.data());
  magicmind::ITensor *begin_tensor = begin->GetOutput(0);
  CHECK_VALID(begin_tensor);

  // create size tensor
  std::vector<int32_t> size_value = {1};
  magicmind::IConstNode *size = network->AddIConstNode(DataType::INT32, Dims({1}), size_value.data());
  magicmind::ITensor *size_tensor = size->GetOutput(0);
  CHECK_VALID(size_tensor);

  // create axis tensor
  std::vector<int32_t> axis_value = {3};
  magicmind::IConstNode *axis = network->AddIConstNode(DataType::INT32, Dims({1}), axis_value.data());
  magicmind::ITensor *axis_tensor = axis->GetOutput(0);
  CHECK_VALID(axis_tensor);

  // add slice node
  ISliceNode *Slice = network->AddISliceNode(input1_tensor, begin_tensor, size_tensor, axis_tensor);

  // mark network output
  for (auto i = 0; i < Slice->GetOutputCount(); i++) {
    auto output_tensor = Slice->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel(
      "slice_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("slice_model"));

  return 0;
}
