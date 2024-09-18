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

  // optional parameter output size
  std::vector<int32_t> output_size_value = {10, 10};
  magicmind::IConstNode *output_size = network->AddIConstNode(DataType::INT32, Dims({2}), output_size_value.data());
  CHECK_VALID(output_size);

  // create interp node
  IInterpNode *Interp = network->AddIInterpNode(input_tensor, output_size->GetOutput(0));

  //using Interp default paramters, you can set each attribute's value.
  //CHECK_STATUS(Interp->SetAlignCorners([value]));
  //CHECK_STATUS(Interp->SetZoomFactor([value]));

  // mark network output
  CHECK_STATUS(network->MarkOutput(Interp->GetOutput(0)));

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("interp_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("interp_model"));

  return 0;
}
