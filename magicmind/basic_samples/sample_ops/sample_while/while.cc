#include <iostream>
#include <memory>
#include <vector>
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
#include "common/macros.h"
#include "common/container.h"
/*
 * To construct network as follow:
 *  input0: tensor()
 *  input1: tensor()
 *  cond = true
 *  begin = 1
 *  step = 1
 *  end = 5
 *  cond' = cond
 *  begin' = begin
 *  input0' = input0
 *  while (cond') {
 *    input0' = input0' + input1;
 *    begin' = begin' + step;
 *    cond' = (begin' < end) ? true : false
 *    list.push_back(input0')
 *  }
 *  while0 = list[-1]
 *  return while0
 */
void ConstructCfgNetwork(magicmind::DataType op_datatype,
                         magicmind::Dims op_dims,
                         const char *model_name) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);
  // create input tensor
  magicmind::ITensor *input_0 = network->AddInput(op_datatype, op_dims);
  CHECK_VALID(input_0);
  magicmind::ITensor *input_1 = network->AddInput(op_datatype, op_dims);
  CHECK_VALID(input_1);
  // create condition
  bool cond_val[1] = {1};
  magicmind::IConstNode *cond_node = network->AddIConstNode(
      magicmind::DataType::BOOL, magicmind::Dims(std::vector<int64_t>()), cond_val);
  CHECK_VALID(cond_node);
  // create loop var
  float begin_val[1] = {1};
  magicmind::IConstNode *begin = network->AddIConstNode(
      magicmind::DataType::FLOAT32, magicmind::Dims(std::vector<int64_t>({1})), begin_val);
  CHECK_VALID(begin);
  float step_val[1] = {1};
  magicmind::IConstNode *step = network->AddIConstNode(
      magicmind::DataType::FLOAT32, magicmind::Dims(std::vector<int64_t>({1})), step_val);
  CHECK_VALID(step);
  float end_val[1] = {5};
  magicmind::IConstNode *end = network->AddIConstNode(
      magicmind::DataType::FLOAT32, magicmind::Dims(std::vector<int64_t>({1})), end_val);
  CHECK_VALID(end);
  // create while body
  magicmind::IWhileNode *while_node = network->AddIWhileNode(cond_node->GetOutput(0));
  CHECK_VALID(while_node);
  // fill in loop
  magicmind::ILoopBody *loop_body = while_node->CreateLoopBody();
  CHECK_VALID(loop_body);
  magicmind::ITensor *recur_input_0 = loop_body->AddRecurrenceTensor(input_0);
  CHECK_VALID(recur_input_0);
  magicmind::IElementwiseNode *add2 =
      loop_body->AddIElementwiseNode(recur_input_0, input_1, magicmind::IElementwise::ADD);
  CHECK_VALID(add2);
  CHECK_STATUS(loop_body->UpdateRecurrenceTensor(recur_input_0, add2->GetOutput(0)));
  magicmind::ITensor *recur_iter = loop_body->AddRecurrenceTensor(begin->GetOutput(0));
  CHECK_VALID(recur_iter);
  magicmind::IElementwiseNode *add3 =
      loop_body->AddIElementwiseNode(recur_iter, step->GetOutput(0), magicmind::IElementwise::ADD);
  CHECK_VALID(add3);
  CHECK_STATUS(loop_body->UpdateRecurrenceTensor(recur_iter, add3->GetOutput(0)));
  magicmind::ITensor *recur_cond = loop_body->AddRecurrenceTensor(cond_node->GetOutput(0));
  CHECK_VALID(recur_cond);
  magicmind::ILogicNode *less =
      loop_body->AddILogicNode(add3->GetOutput(0), end->GetOutput(0), magicmind::ILogic::LT);
  CHECK_VALID(less);
  CHECK_STATUS(loop_body->UpdateRecurrenceTensor(recur_cond, less->GetOutput(0)));
  CHECK_STATUS(loop_body->AddLoopOutput(add2->GetOutput(0)));

  // mark network output
  CHECK_STATUS(network->MarkOutput(while_node->GetOutput(0)));

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel(model_name, network.get()));
  CHECK_VALID(model);
  // save model to file
  CHECK_STATUS(model->SerializeToFile(model_name));
}

int main() {
  std::string model_name = "model_while";
  magicmind::DataType op_datatype = magicmind::DataType::FLOAT32;
  magicmind::Dims op_dims({-1, -1, -1, -1});
  ConstructCfgNetwork(op_datatype, op_dims, model_name.c_str());
  return 0;
}
