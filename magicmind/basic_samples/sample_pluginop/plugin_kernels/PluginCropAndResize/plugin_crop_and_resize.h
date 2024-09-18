/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_CROP_AND_RESIZE_H_
#define PLUGIN_CROP_AND_RESIZE_H_
#include <vector>
#include "mm_plugin.h"
#include "common/macros.h"

namespace magicmind {

Status PluginCropAndResizeDoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> crop_params_shape;
  CHECK_STATUS_RET(context->GetShape("crop_params", &crop_params_shape));
  int64_t total_roi_nums = crop_params_shape[0];
  std::vector<int64_t> input_shape;
  CHECK_STATUS_RET(context->GetShape("input", &input_shape));
  int64_t channel = input_shape[1];
  int64_t d_row, d_col;
  CHECK_STATUS_RET(context->GetAttr("d_row", &d_row));
  CHECK_STATUS_RET(context->GetAttr("d_col", &d_col));
  std::vector<int64_t> output_shape{total_roi_nums, channel, d_row, d_col};
  CHECK_STATUS_RET(context->SetShape("output", output_shape));
  return Status::OK();
}

PLUGIN_REGISTER_OP("PluginCropAndResize")
    .Input("input")
    .TypeConstraint(magicmind::DataType::UINT8)
    .Input("crop_params")
    .TypeConstraint(magicmind::DataType::INT32)
    .Input("roi_nums")
    .TypeConstraint(magicmind::DataType::INT32)
    .Input("pad_values")
    .TypeConstraint(magicmind::DataType::INT32)
    .Output("output")
    .TypeConstraint(magicmind::DataType::UINT8)
    .Param("d_col")
    .Type("int")
    .Param("d_row")
    .Type("int")
    .Param("keep_aspect_ratio")
    .Type("int")
    .Default(1)
    .ShapeFn(PluginCropAndResizeDoShapeInfer);
}  // namespace magicmind

class PluginCropAndResizeKernel : public magicmind::IPluginKernel {
 public:
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  ~PluginCropAndResizeKernel() {}

 private:
  void *input_addr_       = nullptr;
  void *crop_params_addr_ = nullptr;
  void *roi_nums_addr_    = nullptr;
  void *pad_values_addr_  = nullptr;
  void *output_addr_      = nullptr;

  int64_t d_col_;
  int64_t d_row_;
  int64_t keep_aspect_ratio_;

  magicmind::DataType input_dtype_;
  magicmind::DataType crop_params_dtype_;
  magicmind::DataType roi_nums_dtype_;
  magicmind::DataType pad_values_dtype_;
  magicmind::DataType output_dtype_;

  std::vector<int64_t> input_shape_;
  std::vector<int64_t> crop_params_shape_;
  std::vector<int64_t> roi_nums_shape_;
  std::vector<int64_t> pad_values_shape_;
  std::vector<int64_t> output_shape_;

  bool inited_ = false;
  cnrtQueue_t queue_;
};

class PluginCropAndResizeKernelFactory : public magicmind::IPluginKernelFactory {
 public:
  magicmind::IPluginKernel *Create() override { return new PluginCropAndResizeKernel(); }
  ~PluginCropAndResizeKernelFactory() {}
};

namespace magicmind {
PLUGIN_REGISTER_KERNEL(CreatePluginKernelDefBuilder("PluginCropAndResize").DeviceType("MLU"),
                       PluginCropAndResizeKernelFactory);
}  // namespace magicmind
#endif  // PLUGIN_CROP_AND_RESIZE_H_
