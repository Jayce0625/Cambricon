/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <iostream>
#include <string>
#include <cmath>
#include "cn_api.h"
#include "plugin_resize_yuv_to_rgba_kernel.h"
#include "plugin_resize_yuv_to_rgba_macro.h"
#include "plugin_resize_yuv_to_rgba_helper.h"
#include "plugin_resize_yuv_to_rgba.h"

namespace magicmind {

Status PluginResizeYuvToRgbaKernel::SetLocalVar(INodeResource *context) {
  if (!inited_) {
    inited_ = true;
    CHECK_STATUS_RET(context->GetAttr("pad_method", &pad_method_));
    CHECK_STATUS_RET(context->GetAttr("input_format", &input_format_));
    CHECK_STATUS_RET(context->GetAttr("output_format", &output_format_));
    output_format_ += 2;
    CHECK_STATUS_RET(getPixFmtChannelNum((PixelFormat)input_format_, &input_channel_));
    CHECK_STATUS_RET(getPixFmtChannelNum((PixelFormat)output_format_, &output_channel_));
  }

  // Get and check input_y/uv tensors shape
  // Use tensors shape calculate private workspace size.
  std::vector<std::vector<int64_t> > y_tensors_shape;
  std::vector<std::vector<int64_t> > uv_tensors_shape;
  std::vector<std::vector<int64_t> > rgba_tensors_shape;
  CHECK_STATUS_RET(context->GetTensorShape("y_tensors", &y_tensors_shape));
  CHECK_STATUS_RET(context->GetTensorShape("uv_tensors", &uv_tensors_shape));
  CHECK_STATUS_RET(context->GetTensorShape("rgba_tensors", &rgba_tensors_shape));

  // [y_tensors], [uv_tensors], and [rgba_tensors] are TensorList.
  // All elements of these TensorLists can have a 4-dim shape, i.e., NHWC.
  // As a result, batch_num = sum(Ni), where 0 <= i < [y_tensors].size().
  if (y_tensors_shape.size() != uv_tensors_shape.size()) {
    std::string temp =
        "[PluginResizeYuvToRgba] Size of tensorlist [y_tensors] and [uv_tensors] "
        "must be "
        "equal, but now [y_tensors].size() is  " +
        std::to_string(y_tensors_shape.size()) + ", while [uv_tensors].size() is " +
        std::to_string(uv_tensors_shape.size()) + ".";
    Status status(error::Code::INVALID_ARGUMENT, temp);
    return status;
  }
  input_tensor_num_ = y_tensors_shape.size();
  output_tensor_num_ = rgba_tensors_shape.size();
  input_batch_num_vec_.resize(input_tensor_num_);
  output_batch_num_vec_.resize(output_tensor_num_);
  std::vector<DataType> y_tensor_dtypes;
  std::vector<DataType> uv_tensor_dtypes;
  CHECK_STATUS_RET(context->GetTensorDataType("y_tensors", &y_tensor_dtypes));
  CHECK_STATUS_RET(context->GetTensorDataType("uv_tensors", &uv_tensor_dtypes));

  for (int32_t tidx = 0; tidx < input_tensor_num_; tidx++) {
    // check y_dim_num == uv_dim_num == rgba_dim_num == 4
    int32_t y_dim_num = y_tensors_shape[tidx].size();
    int32_t uv_dim_num = uv_tensors_shape[tidx].size();
    if (!(y_dim_num == 4 && uv_dim_num == 4)) {
      std::string temp =
          "[PluginResizeYuvToRgba] The dimension number of corresponding tensors in tensorlists "
          "[y_tensors] and "
          "[uv_tensors] must be equal to 4, but the dimension number of the " +
          std::to_string(tidx) + numberPostfix(tidx) + " tensor of these tensorlists are " +
          std::to_string(y_dim_num) + " and " + std::to_string(uv_dim_num) + " respectively.";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    // check y_dtype == uv_dtype == UINT8
    if (y_tensor_dtypes[tidx] != DataType::UINT8) {
      std::string temp =
          "[PluginResizeYuvToRgba] The datatype of tensor in tensorlists "
          "[y_tensors] must be equal to UINT8, but now got " +
          TypeEnumToString(y_tensor_dtypes[tidx]) + ".";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    if (uv_tensor_dtypes[tidx] != DataType::UINT8) {
      std::string temp =
          "[PluginResizeYuvToRgba] The datatype of tensor in tensorlists "
          "[uv_tensors] must be equal to UINT8, but now got " +
          TypeEnumToString(uv_tensor_dtypes[tidx]) + ".";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    // check y_batch_num == uv_batch_num
    int64_t y_batch_num = y_tensors_shape[tidx][0];
    int64_t uv_batch_num = y_tensors_shape[tidx][0];
    if (y_batch_num != uv_batch_num) {
      std::string temp =
          "[PluginResizeYuvToRgba] The N-dim of the corresponding tensors in tensorlist "
          "[y_tensors] and [uv_tensors] must be equal, but now the N-dim of the " +
          std::to_string(tidx) + numberPostfix(tidx) + " tensor of these tensorlists are " +
          std::to_string(y_batch_num) + " and " + std::to_string(uv_batch_num) + " respectively.";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    batch_num_ += y_batch_num;
    input_batch_num_vec_[tidx] = y_batch_num;
  }

  int64_t output_batch_num = 0;
  std::vector<DataType> rgba_tensor_dtypes;
  CHECK_STATUS_RET(context->GetTensorDataType("rgba_tensors", &rgba_tensor_dtypes));
  for (int32_t tidx = 0; tidx < output_tensor_num_; tidx++) {
    // check rgba_dim_num == 4
    int32_t rgba_dim_num = rgba_tensors_shape[tidx].size();
    if (rgba_dim_num != 4) {
      std::string temp =
          "[PluginResizeYuvToRgba] The dimension number of tensor in tensorlist "
          "[rgba_tensors] must be equal to 4, but the dimension number of the " +
          std::to_string(tidx) + numberPostfix(tidx) + " tensor of the tensorlist is " +
          std::to_string(rgba_dim_num) + ".";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    // check rgba_dtype == UINT8
    if (rgba_tensor_dtypes[tidx] != DataType::UINT8) {
      std::string temp =
          "[PluginResizeYuvToRgba] The datatype of tensor in tensorlists "
          "[rgba_tensors] must be equal to UINT8, but now got " +
          TypeEnumToString(rgba_tensor_dtypes[tidx]) + ".";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    // compute rgba_batch_num
    int64_t rgba_batch_num = rgba_tensors_shape[tidx][0];
    output_batch_num += rgba_batch_num;
    output_batch_num_vec_[tidx] = rgba_batch_num;
  }

  // check output_batch_num == input_batch_num
  if (output_batch_num != batch_num_) {
    std::string temp =
        "[PluginResizeYuvToRgba] The sum of N-dim of tensors, i.e., the number of input and "
        "output images,  in tensorlists [y_tensors], [uv_tensors], and "
        "[rgba_tensors] must be equal, but the number of input images is " +
        std::to_string(batch_num_) + ", while the number of output images is " +
        std::to_string(output_batch_num) + ".";
    Status status(error::Code::INVALID_ARGUMENT, temp);
    return status;
  }

  device_ptrs_size_ = batch_num_ * 3 * sizeof(void *);  // 3 for input_y, input_uv, and output_rgba.
  size_t shapes_vec_size = batch_num_ * 2 * sizeof(int32_t);  // 2 for h and w.

  // can add memory reusing, and MUST check batch_num_
  if (!input_shapes_cpu_) {
    input_shapes_cpu_ = (void *)malloc(shapes_vec_size);
  }
  if (!output_shapes_cpu_) {
    output_shapes_cpu_ = (void *)malloc(shapes_vec_size);
  }

  // check shapes and gather H/W values into input_shapes_cpu_
  int32_t input_shapes_offset = 0;
  for (int32_t tidx = 0; tidx < input_tensor_num_; tidx++) {
    // prepareWorkspace() from resize_convert need int32_t data. Narrow down.
    int32_t input_batch_num = y_tensors_shape[tidx][0];
    int32_t y_h = y_tensors_shape[tidx][1];
    int32_t y_w = y_tensors_shape[tidx][2];
    int32_t y_c = y_tensors_shape[tidx][3];
    int32_t uv_h = uv_tensors_shape[tidx][1];
    int32_t uv_w = uv_tensors_shape[tidx][2];
    int32_t uv_c = uv_tensors_shape[tidx][3];

    // Check y_tensor_size == 2 * uv_tensor_size
    int64_t y_tensor_size = y_h * y_w * y_c;
    int64_t uv_tensor_size = uv_h * uv_w * uv_c;
    if (y_tensor_size != 2 * uv_tensor_size) {
      std::string temp =
          "[PluginResizeYuvToRgba] The size of tensor in tensorlist [y_tensors] must be twice"
          "that of tensor in tensorlist [uv_tensors], but now size of y_tensors[" +
          std::to_string(tidx) + "] is " + std::to_string(y_tensor_size) +
          ", while size of uv_tensors[" + std::to_string(tidx) + "] is " +
          std::to_string(uv_tensor_size) + ".";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    for (int32_t bidx = 0; bidx < input_batch_num; bidx++) {
      ((int32_t *)input_shapes_cpu_)[(input_shapes_offset + bidx) * 2 + 0] = (int32_t)(y_w);
      ((int32_t *)input_shapes_cpu_)[(input_shapes_offset + bidx) * 2 + 1] = (int32_t)(y_h);
    }
    input_shapes_offset += input_batch_num;

    if (y_c != 1) {
      std::string temp =
          "[PluginResizeYuvToRgba] The C-dim of [y_tensors] must be 1, but C-dim of "
          "y_tensors[ " +
          std::to_string(tidx) + "] is " + std::to_string(y_tensors_shape[tidx][3]) + ".";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
  }

  int32_t output_shapes_offset = 0;
  for (int32_t tidx = 0; tidx < output_tensor_num_; tidx++) {
    // prepareWorkspace() from resize_convert need int32_t data. Narrow down.
    int32_t output_batch_num = rgba_tensors_shape[tidx][0];
    int32_t rgba_h = rgba_tensors_shape[tidx][1];
    int32_t rgba_w = rgba_tensors_shape[tidx][2];
    int32_t rgba_c = rgba_tensors_shape[tidx][3];

    for (int32_t bidx = 0; bidx < output_batch_num; bidx++) {
      ((int32_t *)output_shapes_cpu_)[(output_shapes_offset + bidx) * 2 + 0] = rgba_w;
      ((int32_t *)output_shapes_cpu_)[(output_shapes_offset + bidx) * 2 + 1] = rgba_h;
    }
    output_shapes_offset += output_batch_num;

    if (rgba_c != 3 && rgba_c != 4) {
      std::string temp =
          "[PluginResizeYuvToRgba] The C-dim of [rgba_tensors] must be 3 or 4, but C-dim of "
          "[rgba_tensors] batch " +
          std::to_string(tidx) + " is " + std::to_string(rgba_c) + ".";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
  }

  // Check shape of fill_color.
  std::vector<int64_t> fill_color_tensor_shape;
  CHECK_STATUS_RET(context->GetTensorShape("fill_color_tensor", &fill_color_tensor_shape));
  int64_t num_element_of_fill_color = 1;
  for (size_t dim = 0; dim < fill_color_tensor_shape.size(); dim++) {
    num_element_of_fill_color *= fill_color_tensor_shape[dim];
  }
  if (num_element_of_fill_color != rgba_tensors_shape[0][3]) {
    std::string temp =
        "[PluginResizeYuvToRgba] [fill_color] tensor element number must be equal to C-dim of "
        "output tensors, but " +
        std::to_string(num_element_of_fill_color) + " is received";
    Status status(error::Code::INVALID_ARGUMENT, temp);
    return status;
  }
  return Status::OK();
}

size_t PluginResizeYuvToRgbaKernel::GetWorkspaceSize(magicmind::INodeResource *context) {
  magicmind::Status status_input, status_output;
  // Use kNumWidth/HeightLimit and input/output shape to get max-possible workspace_size:
  // 1. convert filter and bias data - constant
  int32_t convert_size = CEIL_ALIGN_SIZE(
      (kNumFilterHeightIn * kNumFilterChIn * kNumFilterChOut + kNumFilterChOut), int16_t);

  // 2. shape data - related to batch_num_
  int32_t shape_size = CEIL_ALIGN_SIZE(batch_num_ * kNumShapeInfo, int32_t);

  // 3. data ptr - related to batch_num_
  //  - 1 mask ptr, 1 copy_filter ptr, 2 weight ptrs, hence 4 per batch in total.
  int32_t ptr_size = CEIL_ALIGN_SIZE(4 * batch_num_, void *);

  // 4. mask data - related to src_w
  //  - use output_shapes and kNumShrink/ExpandLimit to find max-possible size.
  int32_t mask_size = 0;
  for (int bidx = 0; bidx < batch_num_; bidx++) {
    // An accurate mult value is about 1.5 times of dst_w/src_w, here use 2 to ensure
    // workspace is large enough.
    int mult = kNumWidthExpandLimit * 2;
    mask_size +=
        2 * CEIL_ALIGN_SIZE(
                mult * ((int32_t *)input_shapes_cpu_)[2 * bidx + 0] * kNumDefaultChannel, int16_t);
  }

  // 5. weight data - related to output_shapes_cpu_
  // - assume output_roi_w/h == output_shape_w/h
  int32_t weight_size = 0;
  for (int32_t bidx = 0; bidx < batch_num_; bidx++) {
    weight_size +=
        CEIL_ALIGN_SIZE(((int32_t *)output_shapes_cpu_)[2 * bidx + 0] * output_channel_, int16_t);
  }

  // 6. copy filter - related to dst_w / src_w
  int32_t copy_filter_size = 0;
  for (int bidx = 0; bidx < batch_num_; bidx++) {
    // When dst_w / src_w > 64, kernel will use memcpy instead of conv to do src expansion.
    // As a result, maximal value of mult is 64;
    copy_filter_size += CEIL_ALIGN_SIZE(kNumMultLimit * kNumLt * kNumLt, int8_t);
  }

  auxiliary_data_size_ =
      convert_size + shape_size + ptr_size + mask_size + weight_size + convert_size;
  size_t workspace_size = auxiliary_data_size_ + device_ptrs_size_;
  return workspace_size;
}

Status PluginResizeYuvToRgbaKernel::Enqueue(INodeResource *context) {
  cnrtQueue_t queue;
  CHECK_STATUS_RET(context->GetQueue(&queue));
  CHECK_STATUS_RET(context->GetWorkspace(&workspace_in_mlu_));
  CHECK_STATUS_RET(context->GetTensorDataPtr("input_rois", &input_rois_cpu_));
  CHECK_STATUS_RET(context->GetTensorDataPtr("output_rois", &output_rois_cpu_));
  auxiliary_data_size_ = getResizeConvertWorkspaceSize(
      (int32_t *)input_shapes_cpu_, (int32_t *)input_rois_cpu_, (int32_t *)output_shapes_cpu_,
      (int32_t *)output_rois_cpu_, batch_num_, pad_method_, output_channel_,
      (magicmind::PixelFormat)output_format_);
  size_t workspace_size = auxiliary_data_size_ + device_ptrs_size_;
  // Check input_rois/output_rois/input_shapes/output_shapes
  // SetLocalVar() will not be triggered as long as input/output_shapes does not change,
  // but values of input/output_rois_cpu_ can still effect the size/value of workspace.
  // As a result, the fool-proof checks of input/output_rois_cpu values are moved here.
  // Moreover, fool-proof checks will traverse over all rois and shapes values,
  // making it possible to determine whether auxiliary_data can be reused or not.
  // Could add workspace reuse-logic to improve e2e performance.
  CHECK_STATUS_RET(paramCheck((int32_t *)input_shapes_cpu_, (int32_t *)input_rois_cpu_,
                              (int32_t *)output_shapes_cpu_, (int32_t *)output_rois_cpu_,
                              batch_num_, input_channel_, output_channel_));

  if (CN_SUCCESS != cnMallocHost(&workspace_in_cpu_, auxiliary_data_size_ + device_ptrs_size_)) {
    std::string temp = "[PluginResizeYuvToRgba] Malloc temp host memory failed.";
    Status status(error::Code::INVALID_ARGUMENT, temp);
    return status;
  }
  auxiliary_data_in_cpu_ = workspace_in_cpu_;
  device_ptrs_in_cpu_ = (void *)((uint8_t *)auxiliary_data_in_cpu_ + auxiliary_data_size_);
  memset(auxiliary_data_in_cpu_, 0x00, auxiliary_data_size_);

  CHECK_STATUS_RET(prepareWorkspace(
      (int32_t *)input_shapes_cpu_, (int32_t *)input_rois_cpu_, (int32_t *)output_shapes_cpu_,
      (int32_t *)output_rois_cpu_, batch_num_, pad_method_, COLOR_SPACE_BT_601, COLOR_SPACE_BT_601,
      (PixelFormat)input_format_, (PixelFormat)output_format_, workspace_in_mlu_, workspace_in_cpu_,
      yuv_filter_mlu_, yuv_bias_mlu_, shape_data_mlu_, interp_mask_mlu_, interp_weight_mlu_,
      expand_filter_mlu_));

  std::vector<void *> y_mlu_ptrs;
  CHECK_STATUS_RET(context->GetTensorDataPtr("y_tensors", &y_mlu_ptrs));

  std::vector<void *> uv_mlu_ptrs;
  CHECK_STATUS_RET(context->GetTensorDataPtr("uv_tensors", &uv_mlu_ptrs));
  void *fill_color_mlu;
  CHECK_STATUS_RET(context->GetTensorDataPtr("fill_color_tensor", &fill_color_mlu));

  std::vector<void *> rgba_mlu_ptrs;
  CHECK_STATUS_RET(context->GetTensorDataPtr("rgba_tensors", &rgba_mlu_ptrs));

  void *device_ptrs_in_mlu = (void *)((uint8_t *)workspace_in_mlu_ + auxiliary_data_size_);
  int64_t batch_offset = 0;
  for (int32_t tidx = 0; tidx < input_tensor_num_; tidx++) {
    int64_t batch_num = input_batch_num_vec_[tidx];
    for (int64_t bidx = 0; bidx < batch_num; bidx++) {
      int64_t idx = bidx + batch_offset;
      int64_t addr_offset =
          ((int32_t *)input_shapes_cpu_)[2 * idx + 0] *
          ((int32_t *)input_shapes_cpu_)[2 * idx + 1] /* * input_channel(which is 1) */;
      ((uint8_t **)device_ptrs_in_cpu_)[0 * batch_num_ + idx] =
          (uint8_t *)(y_mlu_ptrs[tidx]) + addr_offset * bidx;
      ((uint8_t **)device_ptrs_in_cpu_)[1 * batch_num_ + idx] =
          (uint8_t *)(uv_mlu_ptrs[tidx]) + addr_offset * bidx / 2;
    }
    batch_offset += batch_num;
  }

  batch_offset = 0;
  for (int32_t tidx = 0; tidx < output_tensor_num_; tidx++) {
    int64_t batch_num = output_batch_num_vec_[tidx];
    for (int64_t bidx = 0; bidx < batch_num; bidx++) {
      int64_t idx = bidx + batch_offset;
      int64_t addr_offset = ((int32_t *)output_shapes_cpu_)[2 * idx + 0] *
                            ((int32_t *)output_shapes_cpu_)[2 * idx + 1] * output_channel_;
      ((uint8_t **)device_ptrs_in_cpu_)[2 * batch_num_ + idx] =
          (uint8_t *)(rgba_mlu_ptrs[tidx]) + addr_offset * bidx;
    }
    batch_offset += batch_num;
  }

  CNRT_CHECK(cnrtMemcpyAsync(workspace_in_mlu_, workspace_in_cpu_, workspace_size, queue,
                             cnrtMemcpyHostToDev));

  void **input_y = (void **)device_ptrs_in_mlu;
  void **input_uv = (void **)((uint8_t **)device_ptrs_in_mlu + 1 * batch_num_);
  void **output_rgba = (void **)((uint8_t **)device_ptrs_in_mlu + 2 * batch_num_);

  ResizeYuvToRgbaEnqueue(queue, output_rgba, input_y, input_uv, shape_data_mlu_, interp_mask_mlu_,
                         interp_weight_mlu_, fill_color_mlu, yuv_filter_mlu_, yuv_bias_mlu_,
                         expand_filter_mlu_, (int32_t *)output_rois_cpu_, batch_num_,
                         output_channel_);
  if (workspace_in_cpu_) {
    cnFreeHost(workspace_in_cpu_);
    workspace_in_cpu_ = nullptr;
  }
  return Status::OK();
}

PluginResizeYuvToRgbaKernel::~PluginResizeYuvToRgbaKernel() {
  if (input_shapes_cpu_) {
    free(input_shapes_cpu_);
    input_shapes_cpu_ = nullptr;
  }
  if (output_shapes_cpu_) {
    free(output_shapes_cpu_);
    output_shapes_cpu_ = nullptr;
  }
}

}  // namespace magicmind
