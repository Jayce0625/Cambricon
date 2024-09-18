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
  DataType input_dtype = DataType::FLOAT16;
  DataType output_dtype = DataType::FLOAT32;
  Dims input_dims = Dims({-1, -1, -1, -1});
  // create input tensor
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  ICastNode *Cast = network->AddICastNode(input_tensor, output_dtype);

  CHECK_STATUS(network->MarkOutput(Cast->GetOutput(0)));

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("cast_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("cast_model"));
  return 0;
}
