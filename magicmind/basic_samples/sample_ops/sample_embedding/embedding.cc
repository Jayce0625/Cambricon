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

  DataType weight_dtype = DataType::FLOAT32;
  Dims weight_dims = Dims({10, 4});
  magicmind::ITensor *weight_tensor = network->AddInput(weight_dtype, weight_dims);
  CHECK_VALID(weight_tensor);

  std::vector<int32_t> indices_value = {0, 1, 2, 3};
  magicmind::IConstNode *indices = network->AddIConstNode(DataType::INT32, Dims({4}), indices_value.data());
  CHECK_VALID(indices);

  // optional input padding_idx
  std::vector<int32_t> padding_value = {1};
  magicmind::IConstNode *padding_idx = network->AddIConstNode(DataType::INT32, Dims({1}), padding_value.data());
  CHECK_VALID(padding_idx);

  // create embedding node
  IEmbeddingNode *Embedding = network->AddIEmbeddingNode(weight_tensor, indices->GetOutput(0), padding_idx->GetOutput(0));

  //using Embedding default paramters, you can set each attribute's value.
  CHECK_STATUS(Embedding->SetLayout(Layout::NCHW, Layout::NCHW, Layout::NCHW));

  // mark network output
  for (auto i = 0; i < Embedding->GetOutputCount(); i++) {
    auto output_tensor = Embedding->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("embedding_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("embedding_model"));

  return 0;
}