/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_PLUGINOPS_PLUGIN_SPATIAL_TRANSFORM_OP_PLUGIN_SPATIAL_TRANSFORM_OP_H_
#define SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_PLUGINOPS_PLUGIN_SPATIAL_TRANSFORM_OP_PLUGIN_SPATIAL_TRANSFORM_OP_H_
#include <vector>
#include <string>
#include "mm_plugin.h"  // NOLINT
#include "cnrt.h"              // NOLINT
#include "cnnl.h"              // NOLINT
#include "common/macros.h"
// 1. register custom op
namespace magicmind {

Status PluginSpatialTransformDoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> input_shape;
  CHECK_STATUS_RET(context->GetShape("input", &input_shape));
  CHECK_STATUS_RET(context->SetShape("output", input_shape));
  return Status::OK();
}
PLUGIN_REGISTER_OP("PluginSpatialTransform")
    .Input("input")
    .TypeConstraint("Type1")
    .Input("mat")
    .TypeConstraint("Type2")
    .Input("multable_value")
    .TypeConstraint("Type2")
    .Output("output")
    .TypeConstraint("Type1")
    .Param("Type1")
    .Type("type")
    .Allowed({DataType::FLOAT16, DataType::FLOAT32})
    .Param("Type2")
    .Type("type")
    .Allowed({DataType::FLOAT16, DataType::FLOAT32})
    .Param("op_layout")
    .TypeList("layout")
    .Allowed({Layout::NHWC, Layout::NCHW, Layout::NT, Layout::TN, Layout::ARRAY})
    .ShapeFn(PluginSpatialTransformDoShapeInfer);
}  // namespace magicmind

// 2.create plugin kernel
class PluginSpatialTransformKernel : public magicmind::IPluginKernel {
 public:
  // check kernel param
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  // set plugin workspace
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  ~PluginSpatialTransformKernel();
  PluginSpatialTransformKernel()
      : tensor_names({"input", "mat", "multable_value",
                      "output"}),  // Do something with PluginNode param names.
        cast_types_(4, CNNL_CAST_HALF_TO_FLOAT),
        trans_types_(4, nullptr),
        cnnl_descs_(12, nullptr),
        tensor_element_counts_(4, 0),
        cast_first_flags_(4, 0) {}

 private:
  void *workspace_ = nullptr;
  cnrtQueue_t queue_;

  // The order of following vector is:
  //   input, mat, multable_value, and output;
  // For cnnl_descs_, each input corresponds to 3 elements:
  //   origin_desc, cast_desc, trans_desc
  // Using std::vector makes it easier to duplicate implementation for other op,
  //   especially when cast and trans needed.
  std::vector<std::string> tensor_names;
  std::vector<cnnlCastDataType_t> cast_types_;
  std::vector<cnnlTransposeDescriptor_t> trans_types_;
  std::vector<cnnlTensorDescriptor_t> cnnl_descs_;
  std::vector<uint64_t> tensor_element_counts_;
  std::vector<int32_t> cast_first_flags_;

  int dst_h_            = 0;
  int dst_w_            = 0;
  int src_h_            = 0;
  int src_w_            = 0;
  int c_                = 0;
  int batch_size_       = 0;
  int mat_no_broadcast_ = 0;

  bool inited_ = false;
};

// 3.register kernel
class PluginSpatialTransformKernelFactory : public magicmind::IPluginKernelFactory {
 public:
  // rewrite create
  magicmind::IPluginKernel *Create() override { return new PluginSpatialTransformKernel(); }
  ~PluginSpatialTransformKernelFactory() {}
};

namespace magicmind {
PLUGIN_REGISTER_KERNEL(CreatePluginKernelDefBuilder("PluginSpatialTransform").DeviceType("MLU"),
                       PluginSpatialTransformKernelFactory);
}  // namespace magicmind
#endif  // SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_PLUGINOPS_PLUGIN_SPATIAL_TRANSFORM_OP_PLUGIN_SPATIAL_TRANSFORM_OP_H_
