/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "common/container.h"
#include "common/data.h"
#include "common/macros.h"
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
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

  // create shuffle_channel node
  IShuffleChannelNode *ShuffleChannel =
      network->AddIShuffleChannelNode(input_tensor);
  if(ShuffleChannel) {
    CHECK_STATUS(ShuffleChannel->SetGroup(1));
    CHECK_STATUS(ShuffleChannel->SetAxis(1));
  }

  // mark network output
  for (auto i = 0; i < ShuffleChannel->GetOutputCount(); i++) {
    auto output_tensor = ShuffleChannel->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(
      builder->BuildModel("shuffle_channel_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("shuffle_channel_model"));

  return 0;
}