/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include "cnnl.h"
#include "plugin_spatial_transform_op.h"
#include "kernel_spatial_transform.h"
#include "common/device.h"
namespace {
// Transpose and cast operation is used only in spatial_transform.
// Restrict following helpers and include "cnnl.h" only in this op.
static const uint32_t kNumSpatialTransformTensor =
    4;                                            // [input] , [mat], [multable_value], [output]
static const uint32_t kNumTensorDescElement = 3;  // origin_desc, cast_desc, transpose_desc

#define CHECK_CNNL_RET(status)                                                       \
  do {                                                                               \
    auto __ret = (status);                                                           \
    if (__ret != CNNL_STATUS_SUCCESS) {                                              \
      return magicmind::Status(magicmind::error::Code::INTERNAL, #status " failed"); \
    }                                                                                \
  } while (0)

// MM::DataType to CNNL::DataType
static std::map<magicmind::DataType, cnnlDataType_t> kMMDTypeToCNNLDTypeMap{
    {magicmind::DataType::FLOAT16, CNNL_DTYPE_HALF},
    {magicmind::DataType::FLOAT32, CNNL_DTYPE_FLOAT},
};
static cnnlDataType_t GetCNNLDataType(magicmind::DataType dtype) {
  auto type_iter = kMMDTypeToCNNLDTypeMap.find(dtype);
  if (type_iter == kMMDTypeToCNNLDTypeMap.end()) {
    return CNNL_DTYPE_INVALID;
  }
  return type_iter->second;
}

// MM::Layout to CNNL::Layout
static std::map<magicmind::Layout, cnnlTensorLayout_t> kMMLayoutToCNNLLayoutMap{
    {magicmind::Layout::NHWC, CNNL_LAYOUT_NHWC},  {magicmind::Layout::NCHW, CNNL_LAYOUT_NCHW},
    {magicmind::Layout::NT, CNNL_LAYOUT_ARRAY},   {magicmind::Layout::TN, CNNL_LAYOUT_ARRAY},
    {magicmind::Layout::ARRAY, CNNL_LAYOUT_ARRAY},
};
static cnnlTensorLayout_t GetCNNLLayoutType(magicmind::Layout layout) {
  auto layout_iter = kMMLayoutToCNNLLayoutMap.find(layout);
  if (layout_iter == kMMLayoutToCNNLLayoutMap.end()) {
    return CNNL_LAYOUT_ARRAY;
  }
  return layout_iter->second;
}

// Use high 16-bit store layout_in, use low 16-bit store layout_out
// as an uint32_t value. This value represents different combination
// of layout type.
#define LAYOUT_CASE(IN, OUT)                                 \
  case (static_cast<uint32_t>(magicmind::Layout::IN) << 16 | \
        static_cast<uint32_t>(magicmind::Layout::OUT))

// Use high 16-bit store dtype_in, use low 16-bit store dtype_out
// as an uint32_t value. This value represents different combination
// of dtype.
#define DTYPE_CASE(IN, OUT)                                    \
  case (static_cast<uint32_t>(magicmind::DataType::IN) << 16 | \
        static_cast<uint32_t>(magicmind::DataType::OUT))

// Check if transpose needed when input_layout != expected_layout
// Currently only consider NHWC and NCHW, i.e., check C == 1
static bool TransposeNeeded(magicmind::Layout input_layout,
                            magicmind::Layout expected_layout,
                            const std::vector<int32_t> &input_shape) {
  uint32_t layout_case =
      (static_cast<uint32_t>(input_layout) << 16 | static_cast<uint32_t>(expected_layout));
  switch (layout_case) {
    LAYOUT_CASE(NHWC, NHWC) : { return false; };
    LAYOUT_CASE(NCHW, NCHW) : { return false; };
    LAYOUT_CASE(NHWC, NCHW) : {
      // Should never be triggered.
      if (input_shape.size() != 4) {
        return false;
      }
      return (input_shape[3] != 1);
    };
    LAYOUT_CASE(NCHW, NHWC) : {
      // Should never be triggered.
      if (input_shape.size() != 4) {
        return false;
      }
      return (input_shape[1] != 1);
    };
    default: { return false; }
  }
}

// Get cnnlTransposeDescriptor_t
magicmind::Status GetCNNLTransposeType(cnnlTransposeDescriptor_t *trans_type,
                                       magicmind::Layout trans_in,
                                       magicmind::Layout trans_out) {
  int dims = 8;    // cnnlSetTransposeDesc limitation
  int permute[8];  // cnnlSetTransposeDesc use deepcopy.
  for (int idx = 0; idx < dims; idx++) {
    permute[idx] = idx;
  }
  uint32_t layout_case = (static_cast<uint32_t>(trans_in) << 16 | static_cast<uint32_t>(trans_out));
  switch (layout_case) {
    LAYOUT_CASE(NHWC, NCHW) : {
      // 0 1 2 3 -> 0 3 1 2
      dims = 4;
      permute[1] = 3;
      permute[2] = 1;
      permute[3] = 2;
    };
    break;
    LAYOUT_CASE(NCHW, NHWC) : {
      // 0 1 2 3 -> 0 2 3 1
      dims = 4;
      permute[1] = 2;
      permute[2] = 3;
      permute[3] = 1;
    };
    break;
    default: {
      // Layout combination not supported.
      // Should never be triggered.
      std::string temp =
          "[PluginSpatialTransform Internel] Layout combination is not supported. trans_in layout "
          "is " +
          magicmind::LayoutEnumToString(trans_in) + ", trans_out layout is " +
          magicmind::LayoutEnumToString(trans_out) + ".";
      return magicmind::Status(magicmind::error::Code::UNAVAILABLE, temp);
    }
  }

  // Not likely, but check in case of memory leak
  if (!(*trans_type)) {
    CHECK_CNNL_RET(cnnlCreateTransposeDescriptor(trans_type));
    CHECK_CNNL_RET(cnnlSetTransposeDescriptor((*trans_type), dims, permute));
  }
  return magicmind::Status::OK();
}

static magicmind::Status GetCNNLCastDataType(cnnlCastDataType_t *cast_type,
                                             magicmind::DataType in,
                                             magicmind::DataType out) {
  // Should never be triggereid.
  if (cast_type == NULL) {
    std::string temp = "[PluginSpatialTransform Internel] [" + std::string(__FUNCTION__) +
                       "()] : cast_type pointer must not be NULL.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  uint32_t dtype_case = (static_cast<uint32_t>(in) << 16 | static_cast<uint32_t>(out));
  switch (dtype_case) {
    DTYPE_CASE(FLOAT32, FLOAT16) : { (*cast_type) = CNNL_CAST_FLOAT_TO_HALF; };
    break;
    DTYPE_CASE(FLOAT16, FLOAT32) : { (*cast_type) = CNNL_CAST_HALF_TO_FLOAT; };
    break;
    default: {
      // DType combination not supported.
      // Should never be triggered.
      std::string temp = "[PluginSpatialTransform Internel] [" + std::string(__FUNCTION__) +
                         "()] : DataType combination is not supported. cast_in "
                         "datatype "
                         "is " +
                         magicmind::TypeEnumToString(in) + ", cast_out datatype is " +
                         magicmind::TypeEnumToString(out) + ".";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }
  }
  return magicmind::Status::OK();
}

static magicmind::Status CheckDtypeAndGetDesc(cnnlTensorDescriptor_t *input_desc,
                                              cnnlTensorDescriptor_t *input_cast_desc,
                                              cnnlCastDataType_t *cast_type,
                                              const std::vector<int32_t> &input_shape,
                                              magicmind::DataType input_dtype,
                                              magicmind::DataType expected_dtype,
                                              magicmind::Layout cast_desc_layout,
                                              const std::string &tensor_name,
                                              bool is_output) {
  if (input_dtype != expected_dtype) {
    std::cout << "[PluginSpatialTransoform] [Warning] MLUKernel only supports " +
                     magicmind::TypeEnumToString(expected_dtype) + " datatype for [" + tensor_name +
                     "], but " + magicmind::TypeEnumToString(input_dtype) +
                     " datatype is received. [cnnlCastDataType] operation will be inserted. "
                     "Network accuracy and latency may be differet than expectation."
              << std::endl;
    magicmind::DataType cast_in = (!is_output) ? input_dtype : expected_dtype;
    magicmind::DataType cast_out = (!is_output) ? expected_dtype : input_dtype;

    // Get cnnlCastDataType_t
    CHECK_STATUS_RET(GetCNNLCastDataType(cast_type, cast_in, cast_out));

    // If transpose first, input_desc will be already initialized.
    if (!(*input_desc)) {
      CHECK_CNNL_RET(cnnlCreateTensorDescriptor(input_desc));
      CHECK_CNNL_RET(cnnlSetTensorDescriptor((*input_desc), GetCNNLLayoutType(cast_desc_layout),
                                             GetCNNLDataType(cast_in), input_shape.size(),
                                             input_shape.data()));
    }

    // Initialize input_cast_desc
    // Not likely, but add check in case of memory leak.
    if (!(*input_cast_desc)) {
      CHECK_CNNL_RET(cnnlCreateTensorDescriptor(input_cast_desc));
      CHECK_CNNL_RET(cnnlSetTensorDescriptor(
          (*input_cast_desc), GetCNNLLayoutType(cast_desc_layout), GetCNNLDataType(cast_out),
          input_shape.size(), input_shape.data()));
    }
  }
  return magicmind::Status::OK();
}

static magicmind::Status CheckLayoutAndGetDesc(cnnlTensorDescriptor_t *input_desc,
                                               cnnlTensorDescriptor_t *input_trans_desc,
                                               cnnlTransposeDescriptor_t *trans_type,
                                               const std::vector<int32_t> &input_shape,
                                               magicmind::Layout input_layout,
                                               magicmind::Layout expected_layout,
                                               magicmind::DataType trans_desc_dtype,
                                               const std::string &tensor_name,
                                               bool is_output) {
  if (TransposeNeeded(input_layout, expected_layout, input_shape)) {
    std::cout << "[PluginSpatialTransoform] [Warning] MLUKernel only supports " +
                     magicmind::LayoutEnumToString(expected_layout) + " layout for [" +
                     tensor_name + "], but " + magicmind::LayoutEnumToString(input_layout) +
                     " layout is received. [cnnlTranspose] operation will be inserted. "
                     "Network latency may be differet than expectation."
              << std::endl;
    magicmind::Layout trans_in = (!is_output) ? input_layout : expected_layout;
    magicmind::Layout trans_out = (!is_output) ? expected_layout : input_layout;

    // Get cnnlTransposeDescriptor_t
    CHECK_STATUS_RET(GetCNNLTransposeType(trans_type, trans_in, trans_out));

    // If cast first, input_desc will be already initialized.
    if (!(*input_desc)) {
      CHECK_CNNL_RET(cnnlCreateTensorDescriptor(input_desc));
      CHECK_CNNL_RET(cnnlSetTensorDescriptor((*input_desc), GetCNNLLayoutType(trans_in),
                                             GetCNNLDataType(trans_desc_dtype), input_shape.size(),
                                             input_shape.data()));
    }

    // Initialize input_trans_desc
    // Not likely, but add check in case of memory leak.
    if (!(*input_trans_desc)) {
      CHECK_CNNL_RET(cnnlCreateTensorDescriptor(input_trans_desc));
      CHECK_CNNL_RET(cnnlSetTensorDescriptor((*input_trans_desc), GetCNNLLayoutType(trans_out),
                                             GetCNNLDataType(trans_desc_dtype), input_shape.size(),
                                             input_shape.data()));
    }
  }
  return magicmind::Status::OK();
}

/*
 * 1. Check input_dtype == expected_dtype, if false,
 *   create descs for cnnlCastDataType operation.
 * 2. Check input_layout == expected_layout, if false,
 *   create descs for cnnlTranspose_V2 operation.
 * 3. If both above inserted, compare cast btye_width
 *  to determine cast first or transpose first.
 */
static magicmind::Status CheckAttrAndGetDesc(cnnlTensorDescriptor_t *input_desc,
                                             cnnlTensorDescriptor_t *input_cast_desc,
                                             cnnlTensorDescriptor_t *input_trans_desc,
                                             cnnlCastDataType_t *cast_type,
                                             cnnlTransposeDescriptor_t *trans_type,
                                             int32_t *cast_first,
                                             const std::vector<int32_t> &input_shape,
                                             magicmind::DataType input_dtype,
                                             magicmind::DataType expected_dtype,
                                             magicmind::Layout input_layout,
                                             magicmind::Layout expected_layout,
                                             const std::string &tensor_name,
                                             bool is_output) {
  magicmind::DataType trans_desc_dtype = magicmind::DataType::FLOAT32;
  magicmind::Layout cast_desc_layout = magicmind::Layout::NHWC;
  magicmind::Status status = magicmind::Status::OK();
  if (magicmind::DataTypeSize(input_dtype) >= magicmind::DataTypeSize(expected_dtype)) {
    // datasize is smaller or equivalent, cast first.
    // Need input_desc -> input_cast_desc -> input_trans_desc.
    // If cast is not inserted, expected_dtype == input_dtype,
    //   still works for trans_desc_dtype
    cast_desc_layout = input_layout;
    trans_desc_dtype = expected_dtype;
    (*cast_first) = 1;
  } else {
    // datasize gets larger, transpose first.
    // If transpose is not inserted, expected_layout == input_layout,
    //   still works for cast_desc_layout
    cast_desc_layout = expected_layout;
    trans_desc_dtype = input_dtype;
    (*cast_first) = 0;
  }

  // Currently only works for float16 <-> float32
  if (input_dtype != expected_dtype) {
    CHECK_STATUS_RET(CheckDtypeAndGetDesc(input_desc, input_cast_desc, cast_type, input_shape,
                                          input_dtype, expected_dtype, cast_desc_layout,
                                          tensor_name, is_output));
  }

  // Currently only works for NHWC <-> NCHW
  if (input_layout != expected_layout) {
    CHECK_STATUS_RET(CheckLayoutAndGetDesc(input_desc, input_trans_desc, trans_type, input_shape,
                                           input_layout, expected_layout, trans_desc_dtype,
                                           tensor_name, is_output));
  }
  return magicmind::Status::OK();
}
}  // namespace
// An op "queue" is better, but seems over-designed here,
// and need to implement wrapclass for op and subclasses
// for [cnnlCastDataType], [cnnlTranspose],
// and [spatialTransform]
magicmind::Status PluginSpatialTransformKernel::SetLocalVar(magicmind::INodeResource *context) {
  std::vector<magicmind::Layout> layouts;
  std::vector<magicmind::DataType> tensor_dtypes(kNumSpatialTransformTensor);
  if (!inited_) {
    // Get and check inputs/output layout.
    CHECK_STATUS_RET(context->GetAttr("op_layout", &layouts));
    if (layouts.size() != kNumSpatialTransformTensor) {
      std::string temp =
          "[PluginSpatialTransform] op_layout must have 4 elements which represents layout for "
          "[input], [mat], [multable_value], and [output] accordingly, but now " +
          std::to_string(layouts.size()) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }

    if (layouts[0] != magicmind::Layout::NHWC && layouts[0] != magicmind::Layout::NCHW) {
      std::string temp = "[PluginSpatialTransform] [input] layout must be NCHW or NHWC，but " +
                         magicmind::LayoutEnumToString(layouts[0]) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }

    if (layouts[3] != layouts[0]) {
      std::string temp =
          "[PluginSpatialTransform] [input] layout must be equal to [output] layout, but now "
          "[input] "
          "layout is " +
          magicmind::LayoutEnumToString(layouts[0]) + ", [output] layout is " +
          magicmind::LayoutEnumToString(layouts[3]) + ".";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }

    if (layouts[1] != magicmind::Layout::NHWC && layouts[1] != magicmind::Layout::NCHW &&
        layouts[1] != magicmind::Layout::NT && layouts[1] != magicmind::Layout::TN &&
	layouts[1] != magicmind::Layout::ARRAY) {
      std::string temp =
          "[PluginSpatialTransform] [mat] layout must be NCHW, NHWC, NT, TN, or ARRAY but " +
          magicmind::LayoutEnumToString(layouts[1]) + " is received.";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }

    if (layouts[2] != layouts[1]) {
      std::string temp =
          "[PluginSpatialTransform] [mat] layout must be equal to [multable_value] layout, but now "
          "[mat] layout is " +
          magicmind::LayoutEnumToString(layouts[1]) + ", [multable_value] layout is " +
          magicmind::LayoutEnumToString(layouts[2]) + ".";
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }

    // Get and check inputs/output datatype.
    for (size_t idx = 0; idx < kNumSpatialTransformTensor; idx++) {
      CHECK_STATUS_RET(context->GetTensorDataType(tensor_names[idx], &tensor_dtypes[idx]));
    }

    // Check [input] and [mat] dtype is half and float.
    // Check [output] dtype = [input] dtype and
    //   [multable_value] dtype = [mat] dtype
    for (size_t idx = 0; idx < 2; idx++) {
      if (tensor_dtypes[idx] != magicmind::DataType::FLOAT16 &&
          tensor_dtypes[idx] != magicmind::DataType::FLOAT32) {
        std::string temp = "[PluginSpatialTransform] [" + tensor_names[idx] +
                           "] datatype must be FLOAT16 or FLOAT32，but " +
                           magicmind::TypeEnumToString(tensor_dtypes[idx]) + " is received.";
        magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
        return status;
      }
    }

    // [output] idx 3 vs. [input] idx 0.
    // [multable_value] idx 2 vs. [mat] idx 1.
    //   ==> idx vs. 3 - idx.
    for (size_t idx = 0; idx < 2; idx++) {
      if (tensor_dtypes[idx] != tensor_dtypes[3 - idx]) {
        std::string temp = "[PluginSpatialTransform] [" + tensor_names[idx] +
                           "] datatype must be equal to [" + tensor_names[3 - idx] +
                           "] datatype, but now "
                           "[" +
                           tensor_names[idx] + "] datatype is " +
                           magicmind::TypeEnumToString(tensor_dtypes[idx]) + ", [" +
                           tensor_names[3 - idx] + "] data type is " +
                           magicmind::TypeEnumToString(tensor_dtypes[3 - idx]) + ".";
        magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
        return status;
      }
    }
  }

  // Get and check inputs/output shapes.
  // Tensor shape limitation varies, hard to canonicalize.
  std::vector<std::vector<int64_t> > tensor_shapes(kNumSpatialTransformTensor);
  for (size_t idx = 0; idx < kNumSpatialTransformTensor; idx++) {
    CHECK_STATUS_RET(context->GetTensorShape(tensor_names[idx], &tensor_shapes[idx]));
  }

  // input_shape.dim = output_shape.dim = 4.
  // 4 for NCHW or NHWC
  if (tensor_shapes[0].size() != 4) {
    std::string temp = "[PluginSpatialTransform] [input] shape must have 4 dimension, but a" +
                       std::to_string(tensor_shapes[0].size()) + "-D tensor is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (tensor_shapes[3].size() != 4) {
    std::string temp = "[PluginSpatialTransform] [output] shape must have 4 dimension, but a" +
                       std::to_string(tensor_shapes[3].size()) + "-D tensor is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  // input_shape == output_shape.
  // 4 for NCHW or NHWC
  for (int idx = 0; idx < 4; idx++) {
    if (tensor_shapes[0][idx] != tensor_shapes[3][idx]) {
      std::string temp = "[PluginSpatialTransform] output_shape[" + std::to_string(idx) +
                         "] must be equal to" + "input_shape[" + std::to_string(idx) + "], " +
                         "but now output_shape[" + std::to_string(idx) + "] is " +
                         std::to_string(tensor_shapes[3][idx]) + ", input_shape[" +
                         std::to_string(idx) + "] is " + std::to_string(tensor_shapes[0][idx]);
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
      return status;
    }
  }

  if (layouts[0] == magicmind::Layout::NHWC) {
    batch_size_ = tensor_shapes[0][0];
    src_h_ = tensor_shapes[0][1];
    src_w_ = tensor_shapes[0][2];
    c_ = tensor_shapes[0][3];
  } else {
    batch_size_ = tensor_shapes[0][0];
    c_ = tensor_shapes[0][1];
    src_h_ = tensor_shapes[0][2];
    src_w_ = tensor_shapes[0][3];
  }

  // Can be repalced with src_h/w, placeholder for future op update.
  if (layouts[3] == magicmind::Layout::NHWC) {
    dst_h_ = tensor_shapes[3][1];
    dst_w_ = tensor_shapes[3][2];
  } else {
    dst_h_ = tensor_shapes[3][2];
    dst_w_ = tensor_shapes[3][3];
  }

  if (src_h_ != 40) {
    std::string temp = "[PluginSpatialTransform] [input] H-dim must be equal to 40, but now got " +
                       std::to_string(src_h_);
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (src_w_ != 180) {
    std::string temp = "[PluginSpatialTransform] [input] W-dim must be equal to 180, but now got " +
                       std::to_string(src_w_);
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  // mat shape check, only the first two dim is useful.
  if (tensor_shapes[1].size() < 2) {
    std::string temp =
        "[PluginSpatialTransform] [mat] shape must have at least 2 dimension, but a" +
        std::to_string(tensor_shapes[1].size()) + "-D tensor is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (tensor_shapes[1].size() > 2) {
    for (size_t idx = 2; idx < tensor_shapes[1].size(); idx++) {
      if (tensor_shapes[1][idx] != 1) {
        std::string temp = "[PluginSpatialTransform] mat_shape[" + std::to_string(idx) +
                           "] must be 1, but " + std::to_string(tensor_shapes[1][idx]) +
                           " is received.";
        magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
        return status;
      }
    }
  }

  if (batch_size_ != tensor_shapes[1][0] && 1 != tensor_shapes[1][0]) {
    std::string temp = "[PluginSpatialTransform] mat_shape[0] must be equal to " +
                       std::to_string(batch_size_) + " or 1, but " +
                       std::to_string(tensor_shapes[1][0]) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  mat_no_broadcast_ = (tensor_shapes[1][0] == batch_size_);

  if (6 != tensor_shapes[1][1]) {
    std::string temp = "[PluginSpatialTransform] mat_shape[1] must be equal to 6, but " +
                       std::to_string(tensor_shapes[1][1]) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  // multable_value shape check, only the first two dim is useful.
  if (tensor_shapes[2].size() < 2) {
    std::string temp =
        "[PluginSpatialTransform] [multable_value] shape must have at least 2 dimension, but a" +
        std::to_string(tensor_shapes[2].size()) + "-D tensor is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (tensor_shapes[2].size() > 2) {
    for (size_t idx = 2; idx < tensor_shapes[2].size(); idx++) {
      if (tensor_shapes[1][idx] != 1) {
        std::string temp = "[PluginSpatialTransform] multable_value_shape[" + std::to_string(idx) +
                           "] must be 1, but " + std::to_string(tensor_shapes[2][idx]) +
                           " is received.";
        magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
        return status;
      }
    }
  }

  if (batch_size_ != tensor_shapes[2][0]) {
    std::string temp = "[PluginSpatialTransform] multable_value_shape[0] must be equal to " +
                       std::to_string(batch_size_) + "but " + std::to_string(tensor_shapes[2][0]) +
                       " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (2 != tensor_shapes[2][1]) {
    std::string temp = "[PluginSpatialTransform] multable_value_shape[1] must be equal to 2, but " +
                       std::to_string(tensor_shapes[2][1]) + " is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  // Get tensor element count.
  for (size_t idx = 0; idx < kNumSpatialTransformTensor; idx++) {
    tensor_element_counts_[idx] = 1;
    for (auto dim : tensor_shapes[idx]) {
      tensor_element_counts_[idx] *= dim;
    }
  }

  // SpatialTransformKernel prefered dtypes.
  const std::vector<magicmind::DataType> prefered_dtypes({
      magicmind::DataType::FLOAT16,  // input
      magicmind::DataType::FLOAT32,  // mat
      magicmind::DataType::FLOAT32,  // multable_value
      magicmind::DataType::FLOAT16   // output
  });

  // SpatialTransformKernel prefered layouts.
  const std::vector<magicmind::Layout> prefered_layouts({
      magicmind::Layout::NHWC,  // input
      magicmind::Layout::ARRAY,  // mat
      magicmind::Layout::ARRAY,  // multable_value
      magicmind::Layout::NHWC   // output
  });
  for (size_t idx = 0; idx < kNumSpatialTransformTensor; idx++) {
    // cnnlSetTensorDesc needs [int32_t ] value, but shape_vec is [int64_t], have to narrow down.
    std::vector<int32_t> narrowed_tensor_shape(tensor_shapes[idx].size());
    for (size_t dim = 0; dim < tensor_shapes[idx].size(); dim++) {
      narrowed_tensor_shape[dim] = (int32_t)tensor_shapes[idx][dim];
    }

    // check tensor to insert cast or transpose or null.
    CHECK_STATUS_RET(CheckAttrAndGetDesc(
        &cnnl_descs_[idx * kNumTensorDescElement + 0],
        &cnnl_descs_[idx * kNumTensorDescElement + 1],
        &cnnl_descs_[idx * kNumTensorDescElement + 2], &cast_types_[idx], &trans_types_[idx],
        &cast_first_flags_[idx], narrowed_tensor_shape, tensor_dtypes[idx], prefered_dtypes[idx],
        layouts[idx], prefered_layouts[idx], tensor_names[idx], (idx == 3)));
  }
  return magicmind::Status::OK();
}

size_t PluginSpatialTransformKernel::GetWorkspaceSize(magicmind::INodeResource *context) {
  size_t workspace_size = 0;

  // TODO(): optimize workspace usage
  // idx: 0 ~ 11
  //   idx % kNumTensorDescElement == 1 => cast_desc
  //   idx % kNumTensorDescElement == 2 => trans_desc
  //   idx / kNumTensorDescElement = tensor_id
  for (size_t idx = 0; idx < kNumSpatialTransformTensor * kNumTensorDescElement; idx++) {
    size_t tensor_id = idx / kNumTensorDescElement;
    if (idx % kNumTensorDescElement == 0) {
      // origin_desc use no workspace
      continue;
    }
    if (cnnl_descs_[idx]) {
      workspace_size += tensor_element_counts_[tensor_id] * sizeof(float);
    }
  }
  return workspace_size;
}

magicmind::Status PluginSpatialTransformKernel::Enqueue(magicmind::INodeResource *context) {
  // Get and check runtime contexts
  std::vector<void *> device_ptrs(4, nullptr);
  for (size_t idx = 0; idx < kNumSpatialTransformTensor; idx++) {
    CHECK_STATUS_RET(context->GetTensorDataPtr(tensor_names[idx], &device_ptrs[idx]));
  }
  CHECK_STATUS_RET(context->GetWorkspace(&workspace_));
  CHECK_STATUS_RET(context->GetQueue(&queue_));

  // Determine SpatialTransformKernel input/output address
  // Initialized sp op data_ptrs with PluginKernel data_ptrs
  std::vector<void *> spatial_transform_data_ptr(kNumSpatialTransformTensor);
  spatial_transform_data_ptr = device_ptrs;
  void *free_workspace_head = workspace_;

  // Invoke kernels
  cnnlHandle_t handle;
  CHECK_CNNL_RET(cnnlCreate(&handle));
  CHECK_CNNL_RET(cnnlSetQueue(handle, queue_));

  // An "operation queue" can be implemented to use desc_vec more effectively.
  // workspace is used in an order of input, mat, multable_value, output
  // Check 3 inputs
  for (size_t idx = 0; idx < kNumSpatialTransformTensor - 1; idx++) {
    if (cnnl_descs_[3 * idx + 1] != NULL) {
      // inputs cast inserted.
      // cast_in = device_ptrs[idx].
      // cast_out = spatial_transform_data_ptr = free_workspace_head.
      // update free_workspace_head.
      spatial_transform_data_ptr[idx] = free_workspace_head;
      free_workspace_head =
          (void *)((uint8_t *)free_workspace_head + tensor_element_counts_[idx] * sizeof(float));
      CHECK_CNNL_RET(cnnlCastDataType(handle, cnnl_descs_[3 * idx], device_ptrs[idx],
                                      cast_types_[idx], cnnl_descs_[3 * idx + 1],
                                      spatial_transform_data_ptr[idx]));
    }
  }

  if (cnnl_descs_[10] != NULL) {
    // output cast inserted.
    // cast_in = st_out = free_workspace_head.
    // cast_out = device_ptrs[3];
    spatial_transform_data_ptr[3] = free_workspace_head;
  }

  // SpatialTransoformKernel currently does not use these two variables.
  // Placeholder for future kernel updates
  int data_type_ = 0;
  int cal_type_ = 0;
  SpatialTransformEnqueue(queue_, spatial_transform_data_ptr[3], spatial_transform_data_ptr[0],
                          spatial_transform_data_ptr[1], spatial_transform_data_ptr[2], batch_size_,
                          dst_h_, dst_w_, src_h_, src_w_, c_, data_type_, cal_type_,
                          mat_no_broadcast_);

  if (cnnl_descs_[10] != NULL) {
    // output cast inserted
    // cast_in = st_out = free_workspace_head.
    // cast_out = device_ptrs[3];
    CHECK_CNNL_RET(cnnlCastDataType(handle, cnnl_descs_[9], spatial_transform_data_ptr[3],
                                    cast_types_[3], cnnl_descs_[10], device_ptrs[3]));
  }
  CHECK_CNNL_RET(cnnlDestroy(handle));
  return magicmind::Status::OK();
}

PluginSpatialTransformKernel::~PluginSpatialTransformKernel() {
  for (auto desc : cnnl_descs_) {
    if (desc) {
      cnnlDestroyTensorDescriptor(desc);
      desc = nullptr;
    }
  }

  for (auto trans_type : trans_types_) {
    if (trans_type) {
      cnnlDestroyTransposeDescriptor(trans_type);
      trans_type = nullptr;
    }
  }
}
