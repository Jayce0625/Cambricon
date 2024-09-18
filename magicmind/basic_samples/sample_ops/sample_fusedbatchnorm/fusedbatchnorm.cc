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

  DataType mean_dtype = DataType::FLOAT32;
  Dims mean_dims = Dims({-1});
  magicmind::ITensor *mean_tensor = network->AddInput(mean_dtype, mean_dims);
  CHECK_VALID(mean_tensor);

  DataType variance_dtype = DataType::FLOAT32;
  Dims variance_dims = Dims({-1});
  magicmind::ITensor *variance_tensor = network->AddInput(variance_dtype, variance_dims);
  CHECK_VALID(variance_tensor);

  // optional input
  DataType scale_dtype = DataType::FLOAT32;
  Dims scale_dims = Dims({-1});
  magicmind::ITensor *scale_tensor = network->AddInput(scale_dtype, scale_dims);
  CHECK_VALID(scale_tensor);

  // optional input
  DataType offset_dtype = DataType::FLOAT32;
  Dims offset_dims = Dims({-1});
  magicmind::ITensor *offset_tensor = network->AddInput(offset_dtype, offset_dims);
  CHECK_VALID(offset_tensor);

  // create fused_batch_norm node
  IFusedBatchNormNode *FusedBatchNorm = network->AddIFusedBatchNormNode(input_tensor, mean_tensor, variance_tensor, scale_tensor, offset_tensor);

  //using FusedBatchNorm default paramters, you can set each attribute's value.
  CHECK_STATUS(FusedBatchNorm->SetEpsilon(0.00001));
  CHECK_STATUS(FusedBatchNorm->SetAxis(1));

  // mark network output
  for (auto i = 0; i < FusedBatchNorm->GetOutputCount(); i++) {
    auto output_tensor = FusedBatchNorm->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("fused_batch_norm_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("fused_batch_norm_model"));

  return 0;
}
