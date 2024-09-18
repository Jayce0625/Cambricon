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

  int64_t storage_offset = 0;
  std::vector<int64_t> size = {3, 4, 5};
  std::vector<int64_t> stride = {1, 2, 3};

  // create input tensor
  DataType input1_dtype = DataType::FLOAT32;
  Dims input1_dims = Dims({-1, -1, -1, -1});

  magicmind::ITensor *input1_tensor = network->AddInput(input1_dtype, input1_dims);
  CHECK_VALID(input1_tensor);
  IAsStridedNode *as_strided_op = network->AddIAsStridedNode(input1_tensor, size,
                                                             stride, storage_offset);

  // using Add default paramters, you can set each attribute's value.
  // CHECK_STATUS(as_strided_op->SetStride([value]));
  // CHECK_STATUS(as_strided_op->SetStorageOffset([value]));

  // mark network output
  for (auto i = 0; i < as_strided_op->GetOutputCount(); i++) {
    auto output_tensor = as_strided_op->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("as_strided_model",
                                                                  network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("as_strided_model"));

  return 0;
}
