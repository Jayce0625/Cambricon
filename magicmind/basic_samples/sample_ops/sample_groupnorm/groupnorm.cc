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

  // optional input
  auto filter_dim = magicmind::Dims({6});
  std::vector<float> filter_buffer = GenRand(filter_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *weight = network->AddIConstNode(DataType::FLOAT32, filter_dim, filter_buffer.data());
  CHECK_VALID(weight);

  // optional input
  auto bias_dim = magicmind::Dims({6});
  std::vector<float> bias_buffer = GenRand(bias_dim.GetElementCount(), -1.0f, 1.0f, 0);
  magicmind::IConstNode *bias = network->AddIConstNode(DataType::FLOAT32, bias_dim, bias_buffer.data());
  CHECK_VALID(bias);

  // create groupnorm node
  IGroupNormNode *Groupnorm = network->AddIGroupNormNode(input_tensor,weight->GetOutput(0),bias->GetOutput(0),1,1e-5,1);

  //using Groupnorm default paramters, you can set each attribute's value.
  CHECK_STATUS(Groupnorm->SetNumGroups(3));
  CHECK_STATUS(Groupnorm->SetEps(1e-5));
  CHECK_STATUS(Groupnorm->SetAxis(1));

  // mark network output
  for (auto i = 0; i < Groupnorm->GetOutputCount(); i++) {
    auto output_tensor = Groupnorm->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("groupnorm_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("groupnorm_model"));

  return 0;
}
