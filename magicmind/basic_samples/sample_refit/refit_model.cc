/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "mm_network.h"
#include "common/macros.h"
#include "common/data.h"
#include "common/container.h"
#include "basic_samples/sample_refit/refit_model.h"

using namespace magicmind;

magicmind::IModel *CreateModel(magicmind::IBuilderConfig *builder_cfg) {
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);
  // Create input node
  std::vector<int64_t> input_shape{1, 224, 224, 3};
  ITensor *input_tensor = network->AddInput(DataType::FLOAT32, Dims(input_shape));
  CHECK_VALID(input_tensor);
  // Add conv node and set name to weight tensor, so we can refit them by name later
  std::vector<float> filter_host(32 * 3 * 3 * 3, 1);
  IConstNode *filter_node =
      network->AddIConstNode(DataType::FLOAT32, Dims({32, 3, 3, 3}), filter_host.data());
  CHECK_VALID(filter_node);
  filter_node->GetOutput(0)->SetTensorName("conv0_filter");

  std::vector<float> bias_host(32, 1);
  IConstNode *bias_node = network->AddIConstNode(DataType::FLOAT32, Dims({32}), bias_host.data());
  CHECK_VALID(bias_node);
  bias_node->GetOutput(0)->SetTensorName("conv0_bias");

  IConvNode *conv_node = network->AddIConvNode(input_tensor, filter_node->GetOutput(0),
                                               bias_node->GetOutput(0));  // 1, 224, 224, 32
  CHECK_VALID(conv_node);
  CHECK_STATUS(conv_node->SetLayout(Layout::NHWC, Layout::NHWC, Layout::NHWC));
  CHECK_STATUS(conv_node->SetStride(1, 1));
  CHECK_STATUS(conv_node->SetPaddingMode(IPaddingMode::SAME));
  CHECK_STATUS(conv_node->SetDilation(1, 1));

  // Add RELU node
  IActivationNode *relu =
      network->AddIActivationNode(conv_node->GetOutput(0), IActivation::RELU);  // 1, 224, 224, 32
  CHECK_VALID(relu);
  // Add inner-product node and set name to weight tensor, so we can refit them by name later
  std::vector<float> prod_b_host(224 * 32 * 10, 1);
  IConstNode *prod_b_node =
      network->AddIConstNode(DataType::FLOAT32, Dims({224 * 32, 10}), prod_b_host.data());
  CHECK_VALID(prod_b_node);
  prod_b_node->GetOutput(0)->SetTensorName("prod_b_tensor");

  std::vector<float> prod_bias_host(10, 1);
  IConstNode *prod_bias_node =
      network->AddIConstNode(DataType::FLOAT32, Dims({10}), prod_bias_host.data());
  CHECK_VALID(prod_bias_node);
  prod_bias_node->GetOutput(0)->SetTensorName("prod_bias_tensor");

  IInnerProductNode *inner_product = network->AddIInnerProductNode(
      relu->GetOutput(0), prod_b_node->GetOutput(0), prod_bias_node->GetOutput(0));
  CHECK_VALID(inner_product);
  CHECK_STATUS(inner_product->SetAxis(2));
  CHECK_STATUS(network->MarkOutput(inner_product->GetOutput(0)));  // 224 * 10
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto ret = builder->BuildModel("conv_relu_prod", network.get(), builder_cfg);
  CHECK_VALID(ret);
  return ret;
}
