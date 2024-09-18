/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <dlfcn.h>
#include "mm_builder.h"
#include "mm_network.h"
#include "common/macros.h"
#include "common/param.h"
#include "common/logger.h"
#include "common/container.h"

static void ConstructResizeYuvToRgbaNetwork(magicmind::DataType uint8_dt,
                                            magicmind::DataType int32_dt,
                                            magicmind::Dims fill_color_dim,
                                            std::vector<magicmind::Dims> y_tensors_dim,
                                            std::vector<magicmind::Dims> uv_tensors_dim,
                                            magicmind::Dims roi_tensor_dim,
                                            magicmind::Dims shape_tensor_dim,
                                            std::vector<int> out_roi,
                                            std::vector<int> out_shape,
                                            int64_t input_format,
                                            int64_t output_format,
                                            int64_t pad_method,
                                            const char *model_name) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto config = SUniquePtr<magicmind::IBuilderConfig>(magicmind::CreateIBuilderConfig());
  CHECK_VALID(config);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);
  // Cretae input tensor & add pluginop
  magicmind::TensorMap plugin_inputs;
  std::vector<magicmind::ITensor *> y(y_tensors_dim.size());
  std::vector<magicmind::ITensor *> uv(uv_tensors_dim.size());
  for (size_t tidx = 0; tidx < y.size(); tidx++) {
    y[tidx] = network->AddInput(uint8_dt, y_tensors_dim[tidx]);
    CHECK_VALID(y[tidx]);
  }
  for (size_t tidx = 0; tidx < uv.size(); tidx++) {
    uv[tidx] = network->AddInput(uint8_dt, uv_tensors_dim[tidx]);
    CHECK_VALID(uv[tidx]);
  }
  auto input_rois_tensor = network->AddInput(int32_dt, magicmind::Dims());
  CHECK_VALID(input_rois_tensor);
  // out rois are const for now.
  auto output_rois_const = network->AddIConstNode(int32_dt, roi_tensor_dim, out_roi.data());
  CHECK_VALID(output_rois_const);
  auto output_rois_tensor = output_rois_const->GetOutput(0);
  CHECK_VALID(output_rois_tensor);
  // Static usage for output shape.
  // Use AddInput for dynamic.
  auto output_shapes_const = network->AddIConstNode(int32_dt, shape_tensor_dim, out_shape.data());
  CHECK_VALID(output_shapes_const);
  auto output_shapes_tensor = output_shapes_const->GetOutput(0);
  CHECK_VALID(output_shapes_tensor);
  auto fill_color_tensor = network->AddInput(uint8_dt, fill_color_dim);
  CHECK_VALID(fill_color_tensor);

  std::vector<magicmind::ITensor *> input_rois{input_rois_tensor};
  std::vector<magicmind::ITensor *> output_rois{output_rois_tensor};
  std::vector<magicmind::ITensor *> output_shapes{output_shapes_tensor};
  std::vector<magicmind::ITensor *> fill_color{fill_color_tensor};
  plugin_inputs["y_tensors"] = y;
  plugin_inputs["uv_tensors"] = uv;
  plugin_inputs["input_rois"] = input_rois;
  plugin_inputs["output_rois"] = output_rois;
  plugin_inputs["output_shapes"] = output_shapes;
  plugin_inputs["fill_color_tensor"] = fill_color;
  magicmind::DataTypeMap plugin_outputs_dtype;
  // Only support one output now.
  std::vector<magicmind::DataType> output_dtype_vec{uint8_dt};
  plugin_outputs_dtype["rgba_tensors"] = output_dtype_vec;

  magicmind::IPluginNode *plugin_resize_yuv_to_rgba =
      network->AddIPluginNode("PluginResizeYuvToRgba", plugin_inputs, plugin_outputs_dtype);
  CHECK_VALID(plugin_resize_yuv_to_rgba);
  CHECK_STATUS(plugin_resize_yuv_to_rgba->SetAttr("input_format", input_format));
  CHECK_STATUS(plugin_resize_yuv_to_rgba->SetAttr("output_format", output_format));
  CHECK_STATUS(plugin_resize_yuv_to_rgba->SetAttr("pad_method", pad_method));
  // set outputs nodes
  CHECK_STATUS(network->MarkOutput(plugin_resize_yuv_to_rgba->GetOutput(0)));

  // create model
  auto model =
      SUniquePtr<magicmind::IModel>(builder->BuildModel(model_name, network.get(), config.get()));
  CHECK_VALID(model);
  CHECK_STATUS(model->SerializeToFile(model_name));
}

class YUV2RGBArg : public ArgListBase {
  DECLARE_ARG(plugin_lib, (std::string))->SetDescription("Plugin kernel library.");
  DECLARE_ARG(input_format, (int))
      ->SetDescription("Input format: 1:YUVV420SP_NV12, 2:YUV420SP_NV21")
      ->SetAlternative({"1", "2"});
  DECLARE_ARG(output_format, (int))
      ->SetDescription("Output format: 1:RGB, 2:BGR, 3:RGBA, 4:BGRA, 5:ARGB, 6:ABGR")
      ->SetAlternative({"1", "2", "3", "4", "5", "6"});
  DECLARE_ARG(input_num, (int))->SetDescription("Input data number.");
  DECLARE_ARG(total_batch, (int))->SetDescription("Total batch number.");
  DECLARE_ARG(d_row, (int))->SetDescription("Output hight");
  DECLARE_ARG(d_col, (int))->SetDescription("Output weight");
  DECLARE_ARG(pad_method, (int))
      ->SetDescription(
          "To fill pad around or bottom-right corner, 0: not keep ratio, 1: keep ratio with valid "
          "input in the middle, 2: keep ratio with valid input in the left-top corner")
      ->SetAlternative({"0", "1", "2"});
};

int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  YUV2RGBArg arg_reader;
  arg_reader.ReadIn(args);
  auto plugin_lib = Value(arg_reader.plugin_lib());
  auto kernel_lib = dlopen(plugin_lib.c_str(), RTLD_LAZY);
  if (!kernel_lib) {
    SLOG(ERROR) << "Call dlopen() failed : " << dlerror();
    abort();
  }
  int64_t input_tensor_num = Value(arg_reader.input_num());
  int64_t input_format = Value(arg_reader.input_format());
  int64_t output_format = Value(arg_reader.output_format());
  int64_t pad_method = Value(arg_reader.pad_method());
  int total_batch = Value(arg_reader.total_batch());
  int d_row = Value(arg_reader.d_row());
  int d_col = Value(arg_reader.d_col());
  std::string model_name = "resize_yuv_to_rgba_model";

  magicmind::DataType uint8_dt = magicmind::DataType::UINT8;
  magicmind::DataType int32_dt = magicmind::DataType::INT32;

  auto fill_color_dim = magicmind::Dims({-1});
  auto y_tensor_dim = magicmind::Dims({-1, -1, -1, 1});
  auto uv_tensor_dim = magicmind::Dims({-1, -1, -1, 1});
  auto roi_tensor_dim = magicmind::Dims({total_batch, 4, 1, 1});
  auto shape_tensor_dim = magicmind::Dims({1, 3, 1, 1});
  std::vector<magicmind::Dims> y_tensors_dim(input_tensor_num);
  std::vector<magicmind::Dims> uv_tensors_dim(input_tensor_num);
  for (int32_t tidx = 0; tidx < input_tensor_num; tidx++) {
    y_tensors_dim[tidx] = y_tensor_dim;
    uv_tensors_dim[tidx] = uv_tensor_dim;
  }
  // Static usage for output shape.
  // For dynamic usage, use input instead of const data.
  std::vector<int> out_shapes{total_batch, d_row, d_col};
  // out rois are const for now.
  std::vector<int> out_roi{0, 0, d_col, d_row};
  std::vector<int> output_rois(total_batch * 4);
  int init_idx = 0;
  std::generate(output_rois.begin(), output_rois.end(),
                [out_roi, init_idx]() mutable { return out_roi[init_idx++ % 4]; });

  ConstructResizeYuvToRgbaNetwork(uint8_dt, int32_dt, fill_color_dim, y_tensors_dim, uv_tensors_dim,
                                  roi_tensor_dim, shape_tensor_dim, output_rois, out_shapes,
                                  input_format, output_format, pad_method, model_name.c_str());

  dlclose(kernel_lib);
}
