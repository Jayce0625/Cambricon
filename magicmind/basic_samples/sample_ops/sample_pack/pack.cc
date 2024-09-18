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
  std::vector<int32_t> axis_value = {1};
  magicmind::IConstNode *axis = network->AddIConstNode(DataType::INT32,
                                                       Dims({1}), axis_value.data());
  CHECK_VALID(axis);

  // input is variadic, please giving correct number of input's dtype and dims.
  std::vector<DataType> input_dtype = {DataType::FLOAT32};
  std::vector<Dims> input_dims = {Dims({-1, -1, -1, -1})};
  std::vector<ITensor *> input_tensors;
  for (auto i = 0; i < (int)(input_dtype.size()); i++) {
    ITensor *temp_tensor = network->AddInput(input_dtype[i], input_dims[i]);
    CHECK_VALID(temp_tensor);
    input_tensors.push_back(temp_tensor);
  }

  // create pack node
  IPackNode *Pack = network->AddIPackNode(axis->GetOutput(0), input_tensors);

  // mark network output
  for (auto i = 0; i < Pack->GetOutputCount(); i++) {
    auto output_tensor = Pack->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("pack_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("pack_model"));

  return 0;
}
