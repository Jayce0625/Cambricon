#include <iostream>
#include <memory>
#include <vector>
#include "cnrt.h"
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
#include "common/macros.h"
#include "common/container.h"

/*
 * This example first generates a basic network, then modifies the resulting network in various
 * ways. For example, for a graph with the following structure: Tensor0 Tensor1 |   | add
 *        |
 *       out1 Tensor2
 *         |   |
 *          mul
 *           |
 *          out2 Tensor3
 *             |   |
 *              add2
 *               |
 *              out3
 */
SUniquePtr<magicmind::INetwork> ConstructOriginalNetwork(magicmind::DataType op_datatype,
                                                         magicmind::Dims op_dims) {
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
  magicmind::ITensor *input_2 = network->AddInput(op_datatype, op_dims);
  CHECK_VALID(input_2);
  magicmind::ITensor *input_3 = network->AddInput(op_datatype, op_dims);
  CHECK_VALID(input_3);
  // create add node
  magicmind::IElementwiseNode *add =
      network->AddIElementwiseNode(input_0, input_1, magicmind::IElementwise::ADD);
  CHECK_VALID(add);
  SLOG(INFO) << "Setting add's name as add, return " << add->SetNodeName("add");
  // create mul node
  auto out1 = add->GetOutput(0);
  CHECK_VALID(input_2);
  magicmind::IElementwiseNode *mul =
      network->AddIElementwiseNode(out1, input_2, magicmind::IElementwise::MUL);
  CHECK_VALID(mul);
  SLOG(INFO) << "Setting mul's name as mul, return " << mul->SetNodeName("mul");
  // create add node
  auto out2 = mul->GetOutput(0);
  CHECK_VALID(out2);
  magicmind::IElementwiseNode *add2 =
      network->AddIElementwiseNode(out2, input_3, magicmind::IElementwise::ADD);
  CHECK_VALID(add2);
  SLOG(INFO) << "Setting add2's name as add2, return " << add2->SetNodeName("add2");
  // mark network output
  CHECK_STATUS(network->MarkOutput(add2->GetOutput(0)));
  return network;
}
/*
 * In order to change the mul node to sub node,
 *   1, Find the mul node in the network(for example:by node name);
 *   2, Add new sub node with the input [Tensor2, Tensor3];
 *   3, Change add2's inputs to sub's output[0] instead of Tensor4;
 *   4, Remove mul node.
 *   The resulting network looks like this:
 *   Tensor0 Tensor1
 *       |   |
 *        add
 *         |
 *        out1 Tensor3
 *          |   |
 *           sub
 *            |
 *           out4 Tensor3
 *              |   |
 *               add2
 *                |
 *               out3
 */
void ModifyNetwork(magicmind::DataType op_datatype,
                   magicmind::Dims op_dims,
                   const char *model_name) {
  // init
  auto network = ConstructOriginalNetwork(op_datatype, op_dims);
  CHECK_VALID(network);
  magicmind::ITensor *input_4 = network->AddInput(op_datatype, op_dims);
  CHECK_VALID(input_4);
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  // 1.Find the mul node by name.
  auto mul = network->FindNodeByName("main/mul");
  CHECK_VALID(mul);
  // 2.Add new sub node with the input [out1, Tensor4];
  auto out1 = mul->GetInput(0);
  CHECK_VALID(out1);
  magicmind::IElementwiseNode *sub =
      network->AddIElementwiseNode(out1, input_4, magicmind::IElementwise::SUB);
  CHECK_VALID(sub);
  // 3.Change add2's input[0] to sub's output[0];
  auto add2 = mul->GetSuccessors()[0];
  CHECK_VALID(add2);
  CHECK_STATUS(add2->UpdateInput(0, sub->GetOutput(0)));
  // 4.Remove mul node.
  CHECK_STATUS(network->RemoveNode(mul));
  // 5.Remove Tensor2.
  auto input_2 = network->GetInput(2);
  CHECK_VALID(input_2);
  CHECK_STATUS(network->RemoveInput(input_2));
  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel(model_name, network.get()));
  CHECK_VALID(model);
  // save model to file
  CHECK_STATUS(model->SerializeToFile(model_name));
}

int main() {
  std::string model_name = "model_modify";
  magicmind::DataType op_datatype = magicmind::DataType::FLOAT32;
  magicmind::Dims op_dims({64, 64, 64, 64});
  ModifyNetwork(op_datatype, op_dims, model_name.c_str());
  return 0;
}
