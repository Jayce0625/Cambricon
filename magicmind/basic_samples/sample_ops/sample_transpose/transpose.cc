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

  DataType dim0_dtype = DataType::INT32;
  Dims dim0_dims = Dims({1});
  std::vector<int32_t> axis1_value = {1};
  magicmind::IConstNode *dim0_tensor = network->AddIConstNode(dim0_dtype, dim0_dims,
                                                              axis1_value.data());
  CHECK_VALID(dim0_tensor);

  DataType dim1_dtype = DataType::INT32;
  Dims dim1_dims = Dims({1});
  std::vector<int32_t> axis2_value = {2};
  magicmind::IConstNode *dim1_tensor = network->AddIConstNode(dim1_dtype, dim1_dims,
                                                              axis2_value.data());
  CHECK_VALID(dim1_tensor);

  // create transpose node
  ITransposeNode *Transpose = network->AddITransposeNode(input_tensor, dim0_tensor->GetOutput(0),
                                                         dim1_tensor->GetOutput(0));

  // mark network output
  for (auto i = 0; i < Transpose->GetOutputCount(); i++) {
    auto output_tensor = Transpose->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("transpose_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("transpose_model"));

  return 0;
}
