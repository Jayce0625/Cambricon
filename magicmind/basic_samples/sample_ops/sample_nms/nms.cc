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
  DataType boxes_dtype = DataType::FLOAT32;
  Dims boxes_dims = Dims({10, 4});
  magicmind::ITensor *boxes_tensor = network->AddInput(boxes_dtype, boxes_dims);
  CHECK_VALID(boxes_tensor);

  DataType confidence_dtype = DataType::FLOAT32;
  Dims confidence_dims = Dims({10});
  magicmind::ITensor *confidence_tensor = network->AddInput(confidence_dtype, confidence_dims);
  CHECK_VALID(confidence_tensor);

  // optional input
  std::vector<int32_t> max_output_size_value = {10};
  magicmind::IConstNode *max_output_size = network->AddIConstNode(DataType::INT32, Dims({1}), max_output_size_value.data());
  CHECK_VALID(max_output_size);

  // optional input
  std::vector<float> iou_threshold_value = {0.5};
  magicmind::IConstNode *iou_threshold = network->AddIConstNode(DataType::FLOAT32, Dims({1}), iou_threshold_value.data());
  CHECK_VALID(iou_threshold);

  // optional input
  std::vector<float> scores_threshold_value = {0.5};
  magicmind::IConstNode *scores_threshold = network->AddIConstNode(DataType::FLOAT32, Dims({1}), scores_threshold_value.data());
  CHECK_VALID(scores_threshold);

  // create nms node
  INonMaxSuppressionNode *Nms = network->AddINonMaxSuppressionNode(boxes_tensor,
                                                                   confidence_tensor,
                                                                   max_output_size->GetOutput(0),
                                                                   iou_threshold->GetOutput(0),
                                                                   scores_threshold->GetOutput(0));

  //using Nms default paramters, you can set each attribute's value.
  CHECK_STATUS(Nms->SetInputLayout(0));
  CHECK_STATUS(Nms->SetBoxesCoorinateFormat(1));
  CHECK_STATUS(Nms->SetPadToMaxOutputSize(true));
  CHECK_STATUS(Nms->SetOutputMode(IOutputMode::OUTPUT_TARGET_INDICES));

  // mark network output
  for (auto i = 0; i < Nms->GetOutputCount(); i++) {
    auto output_tensor = Nms->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("nms_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("nms_model"));

  return 0;
}