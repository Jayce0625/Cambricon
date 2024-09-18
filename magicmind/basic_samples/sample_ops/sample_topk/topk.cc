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
  Dims input_dims = Dims({-1, -1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  // DataType k_dtype = DataType::INT32;
  // Dims k_dims = Dims({1});
  // magicmind::ITensor *k_tensor = network->AddInput(k_dtype, k_dims);
  // CHECK_VALID(k_tensor);

  std::vector<int32_t> k_value = {1};
  magicmind::IConstNode *k = network->AddIConstNode(DataType::INT32, Dims({1}), k_value.data());
  CHECK_VALID(k);

  // create topk node
  ITopKNode *Topk = network->AddITopKNode(input_tensor, k->GetOutput(0));

  //using Topk default paramters, you can set each attribute's value.
  CHECK_STATUS(Topk->SetAxis(0));
  //CHECK_STATUS(Topk->SetLargest([value]));
  //CHECK_STATUS(Topk->SetSorted([value]));

  // mark network output
  for (auto i = 0; i < Topk->GetOutputCount(); i++) {
    auto output_tensor = Topk->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("topk_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("topk_model"));

  return 0;
}
