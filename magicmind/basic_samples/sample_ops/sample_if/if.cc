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
/*
 *  input1_tensor = tensor(32,64,64,64)
 *  input2_tensor = tensor(32,64,64,64)
 *  cond_node = true
 *  if (cond_node) {
 *    if_0 = input1_tensor + input2_tensor
 *  } else {
 *    if_0 = input1_tensor - input2_tensor
 *  }
 */
int main(int argc, char **argv) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);

  // create input tensor
  ITensor *input1_tensor = network->AddInput(DataType::FLOAT32, Dims({-1, -1, -1, -1}));
  CHECK_VALID(input1_tensor);

  ITensor *input2_tensor = network->AddInput(DataType::FLOAT32, Dims({-1, -1, -1, -1}));
  CHECK_VALID(input2_tensor);

  bool cond[1] = {1};
  IConstNode *cond_node = network->AddIConstNode(DataType::BOOL, Dims(std::vector<int64_t>()), cond);

  IIfNode *if_node = network->AddIIfNode(cond_node->GetOutput(0));

  // branch then body
  ICondBody *then_body = if_node->CreateThenBody();
  IElementwiseNode *elementwise1 = then_body->AddIElementwiseNode(input1_tensor, input2_tensor, IElementwise::ADD);
  CHECK_STATUS(then_body->AddCondOutput(elementwise1->GetOutput(0)));

  // branch else body
  ICondBody *else_body = if_node->CreateElseBody();
  IElementwiseNode *elementwise2 = else_body->AddIElementwiseNode(input1_tensor, input2_tensor, IElementwise::SUB);
  CHECK_STATUS(else_body->AddCondOutput(elementwise2->GetOutput(0)));

  // link abs after if_node as output
  IAbsNode *abs = network->AddIAbsNode(if_node->GetOutput(0));

  // mark network output
  for (auto i = 0; i < abs->GetOutputCount(); i++) {
    auto output_tensor = abs->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("if_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("if_model"));

  return 0;
}
