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

  DataType crop_size_dtype = DataType::FLOAT32;
  Dims crop_size_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *crop_size_tensor = network->AddInput(crop_size_dtype, crop_size_dims);
  CHECK_VALID(crop_size_tensor);

  // please init size, stride and stroage_offset
  std::vector<int64_t> offset({0});
  int64_t axis = 0;
  float space_number = 0;
  // create crop node
  ICropNode *Crop = network->AddICropNode(input_tensor,crop_size_tensor,offset, axis, space_number);

  //using Crop default paramters, you can set each attribute's value.
  //CHECK_STATUS(Crop->SetOffset([value]));
  //CHECK_STATUS(Crop->SetAxis([value]));
  //CHECK_STATUS(Crop->SetSpaceNumber([value]));

  // mark network output
  for (auto i = 0; i < Crop->GetOutputCount(); i++) {
    auto output_tensor = Crop->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("crop_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("crop_model"));

  return 0;
}
