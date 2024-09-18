/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef RESIZE_YUV_TO_RGBA_H_
#define RESIZE_YUV_TO_RGBA_H_
#include <vector>
#include <string>
#include "mm_plugin.h"
#include "common/macros.h"
#include "plugin_resize_yuv_to_rgba_macro.h"

namespace magicmind {

static inline Status getChnFromPixFmt(int64_t *channel, PixelFormat format) {
  switch (format) {
    case PIX_FMT_GRAY:
    case PIX_FMT_NV12:
    case PIX_FMT_NV21: {
      *channel = 1;
      break;
    };
    case PIX_FMT_RGB:
    case PIX_FMT_BGR: {
      *channel = 3;
      break;
    };
    case PIX_FMT_RGBA:
    case PIX_FMT_BGRA:
    case PIX_FMT_ARGB:
    case PIX_FMT_ABGR: {
      *channel = 4;
      break;
    };
    default: {
      std::string temp = "";
      Status status(error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
  }
  return Status::OK();
}

Status PluginResizeYuvToRgbaDoShapeInfer(IShapeInferResource *context) {
  // Get output_format so that output_channel can be determined.
  int64_t output_format  = 0;
  int64_t output_channel = 4;
  std::vector<std::vector<int64_t> > y_tensors_shape;
  std::vector<std::vector<int64_t> > uv_tensors_shape;
  std::vector<std::vector<int64_t> > output_rgb_tensors_shape;
  CHECK_STATUS_RET(context->GetAttr("output_format", &output_format));
  // PixelFormat 3-8
  output_format += 2;
  CHECK_STATUS_RET(getChnFromPixFmt(&output_channel, (PixelFormat)output_format));

  CHECK_STATUS_RET(context->GetShape("y_tensors", &y_tensors_shape));
  CHECK_STATUS_RET(context->GetShape("uv_tensors", &uv_tensors_shape));
  int64_t input_tensor_num = y_tensors_shape.size();

  void *output_shapes_ptr = nullptr;
  CHECK_STATUS_RET(context->GetTensorDataPtr("output_shapes", &output_shapes_ptr));
  if (!output_shapes_ptr) {
    output_rgb_tensors_shape.resize(input_tensor_num);
    for (int64_t tidx = 0; tidx < input_tensor_num; tidx++) {
      output_rgb_tensors_shape[tidx].resize(4);  // input/output tensors dim must be 4
      output_rgb_tensors_shape[tidx][0] = y_tensors_shape[tidx][0];
      output_rgb_tensors_shape[tidx][1] = -1;
      output_rgb_tensors_shape[tidx][2] = -1;
      output_rgb_tensors_shape[tidx][3] = output_channel;
    }
  } else {
    std::vector<int64_t> output_shapes;
    CHECK_STATUS_RET(context->GetShape("output_shapes", &output_shapes));
    int64_t output_tensor_num = output_shapes[0];
    output_rgb_tensors_shape.resize(output_tensor_num);
    for (int tidx = 0; tidx < output_tensor_num; tidx++) {
      int64_t n = ((int32_t *)output_shapes_ptr)[3 * tidx + 0];
      int64_t h = ((int32_t *)output_shapes_ptr)[3 * tidx + 1];
      int64_t w = ((int32_t *)output_shapes_ptr)[3 * tidx + 2];

      output_rgb_tensors_shape[tidx].resize(4);
      output_rgb_tensors_shape[tidx][0] = n;
      output_rgb_tensors_shape[tidx][1] = h;
      output_rgb_tensors_shape[tidx][2] = w;
      output_rgb_tensors_shape[tidx][3] = output_channel;
    }
  }

  CHECK_STATUS_RET(context->SetShape("rgba_tensors", output_rgb_tensors_shape));
  return Status::OK();
}

PLUGIN_REGISTER_OP("PluginResizeYuvToRgba")
    .Input("y_tensors")
    .TypeConstraint("TensorList")
    .NumberConstraint("num_input_args")
    .Input("uv_tensors")
    .TypeConstraint("TensorList")
    .NumberConstraint("num_input_args")
    .Input("input_rois")
    .TypeConstraint(DataType::INT32)
    .Input("output_rois")
    .TypeConstraint(DataType::INT32)
    .Input("output_shapes")
    .TypeConstraint(DataType::INT32)
    .Input("fill_color_tensor")
    .TypeConstraint(DataType::UINT8)
    .Output("rgba_tensors")
    .NumberConstraint("num_output_args")
    .Param("input_format")
    .Type("int")
    .Allowed({1, 2})
    .Param("output_format")
    .Type("int")
    .Allowed({1, 2, 3, 4, 5, 6})
    .Param("pad_method")
    .Type("int")
    .Default(1)
    .Allowed({0, 1, 2})
    .Param("TensorList")
    .Type("type")
    .Allowed({DataType::UINT8})
    .Param("num_input_args")
    .Type("int")
    .Minimum(1)
    .Param("num_output_args")
    .Type("int")
    .Minimum(1)
    .ShapeFn(PluginResizeYuvToRgbaDoShapeInfer);

class PluginResizeYuvToRgbaKernel : public IPluginKernel {
 public:
  Status SetLocalVar(INodeResource *context) override;
  size_t GetWorkspaceSize(INodeResource *context) override;
  Status Enqueue(INodeResource *context) override;
  ~PluginResizeYuvToRgbaKernel();
  PluginResizeYuvToRgbaKernel() {}

 private:
  // const tensor names
  std::vector<int64_t> input_batch_num_vec_;
  std::vector<int64_t> output_batch_num_vec_;

  // host data ptrs
  void *input_shapes_cpu_  = nullptr;
  void *input_rois_cpu_    = nullptr;
  void *output_shapes_cpu_ = nullptr;
  void *output_rois_cpu_   = nullptr;

  // private workspace memory usage:
  //  - device ptrs:
  //    - [1] shape_data ptr.
  //    - [1] yuv_filter ptr.
  //    - [1] yuv_bias ptr.
  //    - [2 * batch_num] interp_mask ptrs(left and right).
  //    - [batch_num] interp_weight ptrs.
  //    - [batch_num] expand_filter ptrs.
  //  - shape_data: src_w sroi_x sroi_y sroi_w sroi_h dst_w droi_x droi_y droi_w droi_h.
  //  - yuv_filter: const data for yuv420sp to rgb conversion.
  //  - yuv_bias: const data for yuv420sp to rgb conversion.
  //  - interp_mask: a 0/1 mask to collect data that will be used for interp calc.
  //  - interp_weight: weight of interp calc.
  //  - expand_filter: const data for abcd to aaa..bbb..ccc..ddd.. conversion.
  //
  //  Use only one cnrtMemcpyAsync to reduce cost.
  void *workspace_in_cpu_      = nullptr;
  void *workspace_in_mlu_      = nullptr;
  void *auxiliary_data_in_cpu_ = nullptr;
  void *device_ptrs_in_cpu_    = nullptr;

  void *shape_data_mlu_     = nullptr;
  void **interp_mask_mlu_   = nullptr;
  void **interp_weight_mlu_ = nullptr;
  void *yuv_filter_mlu_     = nullptr;
  void *yuv_bias_mlu_       = nullptr;
  void **expand_filter_mlu_ = nullptr;

  int64_t input_tensor_num_   = 0;
  int64_t output_tensor_num_  = 0;
  int64_t batch_num_          = 0;
  int64_t pad_method_         = 0;
  int64_t input_format_       = 1;
  int64_t output_format_      = 3;
  int32_t input_channel_      = 1;
  int32_t output_channel_     = 3;
  size_t auxiliary_data_size_ = 0;
  size_t device_ptrs_size_    = 0;

  bool inited_ = false;
  // cache auxiliary_data to improve e2e performance.
  // bool use_cached_workspace_ = false;
};

class PluginResizeYuvToRgbaKernelFactory : public IPluginKernelFactory {
 public:
  IPluginKernel *Create() override { return new PluginResizeYuvToRgbaKernel(); }
  ~PluginResizeYuvToRgbaKernelFactory() {}
};

PLUGIN_REGISTER_KERNEL(CreatePluginKernelDefBuilder("PluginResizeYuvToRgba")
                           .DeviceType("MLU")
                           .ParamInput("input_rois")
                           .ParamInput("output_rois")
                           .ParamInput("output_shapes"),
                       PluginResizeYuvToRgbaKernelFactory);
}  // namespace magicmind
#endif  // RESIZE_YUV_TO_RGBA_H_
