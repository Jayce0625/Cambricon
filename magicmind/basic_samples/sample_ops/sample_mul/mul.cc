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
  DataType input1_dtype = DataType::FLOAT32;
  Dims input1_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input1_tensor = network->AddInput(input1_dtype, input1_dims);
  CHECK_VALID(input1_tensor);

  DataType input2_dtype = DataType::FLOAT32;
  Dims input2_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input2_tensor = network->AddInput(input2_dtype, input2_dims);
  CHECK_VALID(input2_tensor);

  // create mul node
  IElementwiseNode *Mul = network->AddIElementwiseNode(input1_tensor,input2_tensor,IElementwise::MUL);

  //using Mul default paramters, you can set each attribute's value.
  //CHECK_STATUS(Mul->SetAlpha1([value]));
  //CHECK_STATUS(Mul->SetAlpha2([value]));

  // mark network output
  for (auto i = 0; i < Mul->GetOutputCount(); i++) {
    auto output_tensor = Mul->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("mul_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("mul_model"));

  return 0;
}
