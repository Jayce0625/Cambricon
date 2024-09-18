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
  DataType value_dtype = DataType::FLOAT32;
  Dims value_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *value_tensor = network->AddInput(value_dtype, value_dims);
  CHECK_VALID(value_tensor);

  // create reorg node
  IReorgNode *Reorg = network->AddIReorgNode(value_tensor);

  //using Reorg default paramters, you can set each attribute's value.
  CHECK_STATUS(Reorg->SetStride(2));
  CHECK_STATUS(Reorg->SetReverse(false));
  //CHECK_STATUS(Reorg->SetLayout([value]));

  // mark network output
  for (auto i = 0; i < Reorg->GetOutputCount(); i++) {
    auto output_tensor = Reorg->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("reorg_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("reorg_model"));

  return 0;
}
