/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "mm_builder.h"  // NOLINT
#include "mm_network.h"  // NOLINT
#include "mm_runtime.h"  // NOLINT
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

  // create index tensor
  DataType index_dtype = DataType::INT32;
  Dims index_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *index_tensor = network->AddInput(index_dtype, index_dims);
  CHECK_VALID(index_tensor);

  // create unpool2d node
  magicmind::IUnpool2DNode* Unpool2d = \
            network->AddIUnpool2DNode(input_tensor, index_tensor, magicmind::IUnpoolMode::MAX);

  // mark network output
  for (auto i = 0; i < Unpool2d->GetOutputCount(); i++) {
    auto output_tensor = Unpool2d->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("unpool2d_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("unpool2d_model"));

  return 0;
}
