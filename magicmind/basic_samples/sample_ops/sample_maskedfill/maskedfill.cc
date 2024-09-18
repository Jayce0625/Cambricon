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

  DataType mask_dtype = DataType::BOOL;
  Dims mask_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *mask_tensor = network->AddInput(mask_dtype, mask_dims);
  CHECK_VALID(mask_tensor);
  
  DataType value_dtype = DataType::FLOAT32;
  Dims value_dims = Dims(std::vector<int64_t>({}));
  magicmind::ITensor *value_tensor = network->AddInput(value_dtype, value_dims);
  CHECK_VALID(value_tensor);

  // create maskedfill node
  IMaskedFillNode *MaskedFill = network->AddIMaskedFillNode(input_tensor,mask_tensor,value_tensor);


  // mark network output
  for (auto i = 0; i < MaskedFill->GetOutputCount(); i++) {
    auto output_tensor = MaskedFill->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("maskedfill_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("maskedfill_model"));

  return 0;
}
