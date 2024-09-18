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
  DataType x_dtype = DataType::FLOAT32;
  Dims x_dims = Dims({-1});
  magicmind::ITensor *x_tensor = network->AddInput(x_dtype, x_dims);
  CHECK_VALID(x_tensor);

  DataType y_dtype = DataType::FLOAT32;
  Dims y_dims = Dims({-1});
  magicmind::ITensor *y_tensor = network->AddInput(y_dtype, y_dims);
  CHECK_VALID(y_tensor);

  // create ListDiff node
  IListDiffNode *ListDiff = network->AddIListDiffNode(x_tensor,y_tensor);

  //using ListDiff default paramters, you can set each attribute's value.
  //CHECK_STATUS(ListDiff->SetT([value]));
  //CHECK_STATUS(ListDiff->SetOutIdx([value]));

  // mark network output
  for (auto i = 0; i < ListDiff->GetOutputCount(); i++) {
    auto output_tensor = ListDiff->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("listdiff_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("listdiff_model"));

  return 0;
}