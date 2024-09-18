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
  DataType input_dtype = DataType::INT32;
  Dims input_dims = Dims({-1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  bool mask_value[4] = {true, false, true, false};
  magicmind::IConstNode *mask = network->AddIConstNode(DataType::BOOL, Dims({2, 2}), mask_value);
  CHECK_VALID(mask);

  // create maskedselect node
  IMaskedSelectNode *MaskedSelect = network->AddIMaskedSelectNode(input_tensor,mask->GetOutput(0));

  // mark network output
  for (auto i = 0; i < MaskedSelect->GetOutputCount(); i++) {
    auto output_tensor = MaskedSelect->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("maskedselect_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("maskedselect_model"));

  return 0;
}
