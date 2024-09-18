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

  // create avg_pool2d node
  IAvgPool2DNode *AvgPool2d = network->AddIAvgPool2DNode(input_tensor);

  // using AvgPool2d default paramters, you can set each attribute's value.
  // CHECK_STATUS(AvgPool2d->SetKernel([value]));
  // CHECK_STATUS(AvgPool2d->SetStride([value]));
  // CHECK_STATUS(AvgPool2d->SetPad([value]));
  // CHECK_STATUS(AvgPool2d->SetPaddingMode([value]));
  // CHECK_STATUS(AvgPool2d->SetCeilMode([value]));
  // CHECK_STATUS(AvgPool2d->SetCountIncludePad([value]));
  // CHECK_STATUS(AvgPool2d->SetLayout([value]));

  // mark network output
  for (auto i = 0; i < AvgPool2d->GetOutputCount(); i++) {
    auto output_tensor = AvgPool2d->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("avg_pool2d_model",
                                                                  network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("avg_pool2d_model"));

  return 0;
}
