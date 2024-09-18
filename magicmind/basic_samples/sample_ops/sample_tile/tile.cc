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

  DataType multiples_dtype = DataType::INT32;
  Dims multiples_dims = Dims({4});
  std::vector<int32_t> axis_value = {2, 2, 2, 2};
  magicmind::IConstNode *multi = network->AddIConstNode(multiples_dtype, multiples_dims,
                                                       axis_value.data());
  CHECK_VALID(multi);

  // create tile node
  ITileNode *Tile = network->AddITileNode(input_tensor, multi->GetOutput(0));

  // using Tile default paramters, you can set each attribute's value.
  // CHECK_STATUS(Tile->SetAxis([value]));
  // CHECK_STATUS(Tile->SetUseAxis([value]));

  // mark network output
  for (auto i = 0; i < Tile->GetOutputCount(); i++) {
    auto output_tensor = Tile->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("tile_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("tile_model"));

  return 0;
}
