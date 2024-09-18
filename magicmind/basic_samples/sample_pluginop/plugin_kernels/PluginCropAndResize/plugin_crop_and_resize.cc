/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <iostream>
#include <string>
#include "plugin_crop_and_resize_kernel.h"
#include "plugin_crop_and_resize.h"

magicmind::Status PluginCropAndResizeKernel::SetLocalVar(magicmind::INodeResource *context) {
  if (!inited_) {
    inited_ = true;
    CHECK_STATUS_RET(context->GetTensorDataType("input", &input_dtype_));
    CHECK_STATUS_RET(context->GetTensorDataType("crop_params", &crop_params_dtype_));
    CHECK_STATUS_RET(context->GetTensorDataType("roi_nums", &roi_nums_dtype_));
    CHECK_STATUS_RET(context->GetTensorDataType("pad_values", &pad_values_dtype_));
    CHECK_STATUS_RET(context->GetTensorDataType("output", &output_dtype_));
    CHECK_STATUS_RET(context->GetAttr("d_col", &d_col_));
    CHECK_STATUS_RET(context->GetAttr("d_row", &d_row_));
    CHECK_STATUS_RET(context->GetAttr("keep_aspect_ratio", &keep_aspect_ratio_));

    if (input_dtype_ != magicmind::DataType::UINT8) {
      std::string temp = "Input data type is invalid，should be FLOAT16，but " +
                         magicmind::TypeEnumToString(input_dtype_) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }
    if (crop_params_dtype_ != magicmind::DataType::INT32) {
      std::string temp = "crop_params data type is invalid，should be INT32，but " +
                         magicmind::TypeEnumToString(crop_params_dtype_) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }
    if (roi_nums_dtype_ != magicmind::DataType::INT32) {
      std::string temp = "roi_nums data type is invalid，should be INT32，but " +
                         magicmind::TypeEnumToString(roi_nums_dtype_) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }
    if (pad_values_dtype_ != magicmind::DataType::INT32) {
      std::string temp = "pad_values data type is invalid，should be INT32，but " +
                         magicmind::TypeEnumToString(pad_values_dtype_) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }
    if (output_dtype_ != magicmind::DataType::UINT8) {
      std::string temp = "output data type is invalid，should be FLOAT16，but " +
                         magicmind::TypeEnumToString(output_dtype_) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }
  }

  CHECK_STATUS_RET(context->GetTensorShape("input", &input_shape_));
  CHECK_STATUS_RET(context->GetTensorShape("crop_params", &crop_params_shape_));
  CHECK_STATUS_RET(context->GetTensorShape("roi_nums", &roi_nums_shape_));
  CHECK_STATUS_RET(context->GetTensorShape("pad_values", &pad_values_shape_));
  CHECK_STATUS_RET(context->GetTensorShape("output", &output_shape_));

  if (input_shape_.size() != 4) {
    std::string temp = "input must have 4 dimentions, but " + std::to_string(input_shape_.size()) +
                       " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (crop_params_shape_.size() != 2) {
    std::string temp = "crop_params must have 2 dimentions, but " +
                       std::to_string(crop_params_shape_.size()) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (roi_nums_shape_.size() != 1) {
    std::string temp = "roi_nums must have 1 dimentions, but " +
                       std::to_string(roi_nums_shape_.size()) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (pad_values_shape_.size() != 1) {
    std::string temp = "pad_values must have 1 dimentions, but " +
                       std::to_string(pad_values_shape_.size()) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (output_shape_.size() != 4) {
    std::string temp = "output must have 4 dimentions, but " +
                       std::to_string(output_shape_.size()) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (input_shape_[0] != roi_nums_shape_[0]) {
    std::string temp = "input_shape[0] must be equal to roi_nums_shape[0], but input_shape[0] " +
                       std::to_string(input_shape_[0]) + ", and roi_nums_shape[0] " +
                       std::to_string(roi_nums_shape_[0]) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (input_shape_[1] != output_shape_[1]) {
    std::string temp = "input_shape[1] must be equal to output_shape[1], but input_shape[1] " +
                       std::to_string(input_shape_[1]) + ", and output_shape[1] " +
                       std::to_string(output_shape_[1]) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (crop_params_shape_[0] != output_shape_[0]) {
    std::string temp =
        "crop_params_shape[0] must be equal to output_shape[0], but crop_params_shape[0] " +
        std::to_string(crop_params_shape_[0]) + ", and output_shape[0] " +
        std::to_string(output_shape_[0]) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (crop_params_shape_[1] != 4) {
    std::string temp = "crop_params_shape[1] must be equal to 4, but crop_params_shape[1] " +
                       std::to_string(crop_params_shape_[1]) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (pad_values_shape_[0] != 4) {
    std::string temp = "pad_values_shape[0] must be equal to 4, but pad_values_shape[0] " +
                       std::to_string(pad_values_shape_[0]) + " is received";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  return magicmind::Status::OK();
}

size_t PluginCropAndResizeKernel::GetWorkspaceSize(magicmind::INodeResource *context) {
  size_t workspace_size = 0;
  return workspace_size;
}

magicmind::Status PluginCropAndResizeKernel::Enqueue(magicmind::INodeResource *context) {
  CHECK_STATUS_RET(context->GetTensorDataPtr("input", &input_addr_));
  CHECK_STATUS_RET(context->GetTensorDataPtr("crop_params", &crop_params_addr_));
  CHECK_STATUS_RET(context->GetTensorDataPtr("roi_nums", &roi_nums_addr_));
  CHECK_STATUS_RET(context->GetTensorDataPtr("pad_values", &pad_values_addr_));
  CHECK_STATUS_RET(context->GetTensorDataPtr("output", &output_addr_));

  int batch_size = input_shape_[0];
  int s_row = input_shape_[2];
  int s_col = input_shape_[3];

  CHECK_STATUS_RET(context->GetQueue(&queue_));
  CropAndResizeEnqueue(queue_, output_addr_, input_addr_, crop_params_addr_, roi_nums_addr_,
                       pad_values_addr_, s_col, s_row, d_col_, d_row_, 1, 1, batch_size,
                       keep_aspect_ratio_);
  return magicmind::Status::OK();
}
