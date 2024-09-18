/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <ctime>
#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <cstring>
#include "cnrt.h"        // NOLINT
#include "mm_builder.h"  // NOLINT
#include "mm_network.h"  // NOLINT
#include "mm_runtime.h"  // NOLINT
#include "mm_common.h"   // NOLINT
#include "common/macros.h"
#include "common/data.h"
#include "common/container.h"
/*
 * To construct network as follow:
 *     input/bias/filter
 *            |
 *       Convolution(input precision: qint8, qint8, float32, output precision: float32)
 *            |
 *          ReLU(input precision: float16, output precision: float16)
 *            |
 *       DepthwiseConv(input precision: qint8, qint8, output precision: float16)
 *            |
 *          ReLU(input precision: float32, output precision: float32)
 *            |
 *       Convolution(input precision: qint8, qint8, output precision: float32)
 *            |
 *          ReLU(input precision: float32, output precision: float32)
 *            |
 *          output
 */
magicmind::INetwork *ConstructNetwork(magicmind::Dims input_dim,
                                      magicmind::Dims filter_dim,
                                      magicmind::Dims bias_dim) {
  auto network = magicmind::CreateINetwork();
  CHECK_VALID(network);
  // --------------------- create conv and relu ---------------------
  // create input
  auto input_tensor = network->AddInput(magicmind::DataType::FLOAT32, input_dim);
  CHECK_VALID(input_tensor);
  // create filter
  auto filter_tensor = network->AddInput(magicmind::DataType::QINT8, filter_dim);
  CHECK_VALID(filter_tensor);
  // create bias
  std::vector<float> bias_float = GenRand(bias_dim.GetElementCount(), -1.0f, 1.0f, 0);
  auto bias = network->AddIConstNode(magicmind::DataType::FLOAT32, bias_dim, bias_float.data());
  CHECK_VALID(bias);
  auto bias_tensor = bias->GetOutput(0);
  CHECK_VALID(bias_tensor);
  // create conv
  auto conv = network->AddIConvNode(input_tensor, filter_tensor, bias_tensor);
  CHECK_VALID(conv);
  CHECK_STATUS(conv->SetStride(2, 2));
  CHECK_STATUS(conv->SetPad(0, 0, 0, 0));
  CHECK_STATUS(conv->SetDilation(1, 1));
  CHECK_STATUS(
      conv->SetLayout(magicmind::Layout::NHWC, magicmind::Layout::NHWC, magicmind::Layout::NHWC));
  CHECK_STATUS(conv->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));
  auto conv_output = conv->GetOutput(0);
  CHECK_VALID(conv_output);
  // create relu
  auto relu = network->AddIActivationNode(conv_output, magicmind::IActivation::RELU);
  CHECK_VALID(relu);

  // -------------------- create dwconv and relu --------------------
  // create input
  auto dwconv_input_tensor = relu->GetOutput(0);
  CHECK_VALID(dwconv_input_tensor);
  // create filter
  auto dwconv_filter_dim = magicmind::Dims({2, 4, 4, 2});
  std::vector<float> dwconv_filter_float =
      GenRand(dwconv_filter_dim.GetElementCount(), -1.0f, 1.0f, 0);
  std::vector<uint16_t> dwconv_filter_half(dwconv_filter_dim.GetElementCount());
  CHECK_STATUS(NormalCast(dwconv_filter_half.data(), magicmind::DataType::FLOAT16,
                          dwconv_filter_float.data(), magicmind::DataType::FLOAT32,
                          dwconv_filter_dim.GetElementCount(), false /* no saturation */));
  auto dwconv_filter = network->AddIConstNode(magicmind::DataType::FLOAT16, dwconv_filter_dim,
                                              dwconv_filter_half.data());
  CHECK_VALID(dwconv_filter);
  auto dwconv_filter_tensor = dwconv_filter->GetOutput(0);
  CHECK_VALID(dwconv_filter_tensor);
  // create dwconv
  auto dwconv =
      network->AddIConvDepthwiseNode(dwconv_input_tensor, dwconv_filter_tensor, nullptr /*bias*/);
  CHECK_VALID(dwconv);
  CHECK_STATUS(
      dwconv->SetLayout(magicmind::Layout::NHWC, magicmind::Layout::NHWC, magicmind::Layout::NHWC));
  CHECK_STATUS(dwconv->SetPaddingMode(magicmind::IPaddingMode::SAME));
  auto dwconv_output = dwconv->GetOutput(0);
  CHECK_VALID(dwconv_output);
  // create relu
  auto relu2 = network->AddIActivationNode(dwconv_output, magicmind::IActivation::RELU);
  CHECK_VALID(relu2);

  // --------------------- create conv2 and relu ---------------------
  // create input
  auto conv2_input_tensor = relu2->GetOutput(0);
  CHECK_VALID(conv2_input_tensor);
  // create filter
  auto conv2_filter_dim = magicmind::Dims({2, 3, 3, 4});
  std::vector<float> conv2_filter_float =
      GenRand(conv2_filter_dim.GetElementCount(), -2.0f, 2.0f, 0);
  auto conv2_filter = network->AddIConstNode(magicmind::DataType::FLOAT32, conv2_filter_dim,
                                             conv2_filter_float.data());
  CHECK_VALID(conv2_filter);
  auto conv2_filter_tensor = conv2_filter->GetOutput(0);
  CHECK_VALID(conv2_filter_tensor);
  // create conv
  auto conv2 = network->AddIConvNode(conv2_input_tensor, conv2_filter_tensor, nullptr /*bias*/);
  CHECK_VALID(conv2);
  CHECK_STATUS(
      conv2->SetLayout(magicmind::Layout::NHWC, magicmind::Layout::NHWC, magicmind::Layout::NHWC));
  CHECK_STATUS(conv2->SetPaddingMode(magicmind::IPaddingMode::SAME));
  auto conv2_output = conv2->GetOutput(0);
  CHECK_VALID(conv2_output);
  // create relu
  auto relu3 = network->AddIActivationNode(conv2_output, magicmind::IActivation::RELU);
  CHECK_VALID(relu3);

  // set outputs nodes
  CHECK_STATUS(network->MarkOutput(relu3->GetOutput(0)));
  return network;
}

void SetNetworkRanges(magicmind::INetwork *network) {
  // get first conv node
  auto conv_node = network->FindNodeByName("main/mm.conv2d");
  CHECK_VALID(conv_node);
  // set range of filter
  // transfer uniform quantization parameters to range
  magicmind::Range filter_range =  // range is {-1.0f, 1.0f}
      magicmind::UniformQuantParamToRangeWithQuantAlg({0.00787402 /*scale*/, 0 /*offset*/},
                                                      8 /*bitwidth*/, "symmetric");
  // Although the global config in function SetExpectedPrecision is per_axis, but you can still set
  // a range to make ops quantized by per_tensor.
  CHECK_STATUS(conv_node->GetInput(1)->SetDynamicRange(filter_range, false));

  // get dwconv node
  auto dwconv_node = network->FindNodeByName("main/mm.conv_depthwise2d");
  CHECK_VALID(dwconv_node);
  // set range of input
  std::vector<magicmind::Range> input_ranges = {{-1.0f /*min*/, 1.0f /*max*/}, {-1.0f, 1.0f}};
  CHECK_STATUS(dwconv_node->GetInput(0)->SetDynamicRangePerAxis(input_ranges, false));

  // get second conv node
  auto conv2_node = network->FindNodeByName("main/mm.conv2d-1");
  CHECK_VALID(conv2_node);
  // set range of input
  CHECK_STATUS(conv2_node->GetInput(0)->SetDynamicRange({-1.0f, 1.0f}, false));
  // set range of filter
  // transfer normal quantization parameters to uniform quantization parameters
  auto uni_quant_param =
      magicmind::NormalToUniformCast({-13 /*pos*/, 1.99994 /*scale*/, 0 /*offset*/});
  magicmind::Range conv2_filter_range =
      magicmind::UniformQuantParamToRangeWithQuantAlg(uni_quant_param, 16, "symmetric");
  CHECK_STATUS(conv2_node->GetInput(1)->SetDynamicRangePerAxis(
      {conv2_filter_range, conv2_filter_range}, false));
}

magicmind::IBuilderConfig *SetExpectedPrecision(magicmind::INetwork *parameterized_network) {
  auto config = magicmind::CreateIBuilderConfig();
  CHECK_VALID(config);
  // global config, refer to "Cambricon-MagicMind-User-Guide-CN"
  CHECK_STATUS(config->ParseFromString(
      R"({"precision_config": {"precision_mode": "qint8_mixed_float32"}})"));
  CHECK_STATUS(config->ParseFromString(
      R"({"precision_config": {"weight_quant_granularity": "per_axis"}})"));
  CHECK_STATUS(
      config->ParseFromString(R"({"precision_config": {"activation_quant_algo": "asymmetric"}})"));
  // Input and filter of convDepthwise are quantified by per_axis.
  CHECK_STATUS(config->ParseFromString(
      R"({"custom_nodes" : { "IConvDepthwiseNode" : { "input" : { "0" : { "quant_granularity" : "per_axis" }} } }})"));
  CHECK_STATUS(config->ParseFromString(
      R"({"custom_nodes" : { "IConvDepthwiseNode" : { "input" : { "1" : { "quant_granularity" : "per_axis" }} } }})"));
  // shape of input is immutable
  CHECK_STATUS(config->ParseFromString(R"({"graph_shape_mutable": false})"));

  // get first conv node
  auto conv_node = parameterized_network->FindNodeByName("main/mm.conv2d");
  CHECK_VALID(conv_node);
  // set precision and normal quantization parameters of input
  magicmind::Range input_range = {-2.0f, 1.0f};
  // transfer range to uniform quantization parameters
  auto uni_quant_param = RangeToUniformQuantParamWithQuantAlgV2(
      input_range, 8, "asymmetric", magicmind::RoundingMode::ROUND_HALF_TO_EVEN);
  // transfer uniform quantization parameters to normal quantization parameters
  auto nor_quant_param = UniformToNormalCast(uni_quant_param);
  // Set the preferred computational precision and quantization parameters
  CHECK_STATUS(conv_node->SetPrecisionWithNormalParam(0, magicmind::DataType::QINT8,
                                                      nor_quant_param, false));
  // set precision of filter and output
  CHECK_STATUS(conv_node->SetPrecision(1, magicmind::DataType::QINT8));
  CHECK_STATUS(conv_node->SetOutputType(0, magicmind::DataType::FLOAT32));

  // get first relu node
  auto relu_node = parameterized_network->FindNodeByName("main/mm.relu");
  CHECK_VALID(relu_node);
  // set precision of input and output
  CHECK_STATUS(relu_node->SetPrecision(0, magicmind::DataType::FLOAT16));
  CHECK_STATUS(relu_node->SetOutputType(0, magicmind::DataType::FLOAT16));

  // get dwconv node
  auto dwconv_node = parameterized_network->FindNodeByName("main/mm.conv_depthwise2d");
  CHECK_VALID(dwconv_node);
  // set precision of input
  CHECK_STATUS(dwconv_node->SetPrecision(0, magicmind::DataType::QINT8));
  // set precision and normal quantization parameters of filter
  std::vector<magicmind::Range> filter_ranges = {
      {-1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, 1.0f}};
  std::vector<magicmind::QuantizationParam> nor_quant_params;
  for (auto range : filter_ranges) {
    auto uni_quant_param = magicmind::RangeToUniformQuantParamWithQuantAlgV2(
        range, 8, "symmetric", magicmind::RoundingMode::ROUND_HALF_TO_EVEN);
    nor_quant_params.push_back(UniformToNormalCast(uni_quant_param));
  }
  CHECK_STATUS(dwconv_node->SetPrecisionWithNormalParamPerAxis(1, magicmind::DataType::QINT8,
                                                               nor_quant_params, false));
  // set precision of output
  CHECK_STATUS(dwconv_node->SetOutputType(0, magicmind::DataType::FLOAT16));
  return config;
}

void GenerateModel(magicmind::INetwork *parameterized_network,
                   magicmind::IBuilderConfig *config,
                   const char *model_name) {
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  // create model
  auto model =
      SUniquePtr<magicmind::IModel>(builder->BuildModel(model_name, parameterized_network, config));
  CHECK_VALID(model);
  CHECK_STATUS(model->SerializeToFile(model_name));
}

int main() {
  std::string model_name = "model_quantization";
  auto input_dim = magicmind::Dims({1, 224, 224, 3});
  auto filter_dim = magicmind::Dims({2, 5, 5, 3});
  auto bias_dim = magicmind::Dims({2});
  // construct network
  auto network = ConstructNetwork(input_dim, filter_dim, bias_dim);
  // Here use SetDynamicRange or SetDynamicRangePerAxis to set ranges manually, if you use
  // calibration to set ranges, network may be changed, so you should use FindNodeByName to get
  // node.
  SetNetworkRanges(network);
  // Set global configurations for precision_config and custom_nodes, and use api to set precision
  // for some nodes.
  auto config = SetExpectedPrecision(network);
  // build and save model
  GenerateModel(network, config, model_name.c_str());
  // free config and network
  config->Destroy();
  network->Destroy();
  return 0;
}