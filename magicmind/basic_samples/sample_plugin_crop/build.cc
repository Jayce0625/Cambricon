/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <dlfcn.h>
#include "mm_builder.h"
#include "mm_network.h"
#include "common/macros.h"
#include "common/logger.h"
#include "common/container.h"

void ConstructCropAndResizeNetwork(magicmind::DataType uint8_dt,
                                   magicmind::DataType int32_dt,
                                   magicmind::Dims input_dim,
                                   magicmind::Dims crop_params_dim,
                                   magicmind::Dims roi_nums_dim,
                                   magicmind::Dims pad_values_dim,
                                   int64_t d_col,
                                   int64_t d_row,
                                   int64_t keep_aspect_ratio,
                                   const char *model_name) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto config = SUniquePtr<magicmind::IBuilderConfig>(magicmind::CreateIBuilderConfig());
  auto json_string =
        std::string(
            R"({"crop_config": {
                   "mtp_372": {
                      "plugin": {
                        "lib_path": "./libmagicmind_plugin_static.a"
                       }
                   }
                }})");
  CHECK_STATUS(config->ParseFromString(json_string));
  CHECK_VALID(config);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);
  // create input tensor
  auto input_tensor = network->AddInput(uint8_dt, input_dim);
  CHECK_VALID(input_tensor);
  auto crop_params_tensor = network->AddInput(int32_dt, crop_params_dim);
  CHECK_VALID(crop_params_tensor);
  auto roi_nums_tensor = network->AddInput(int32_dt, roi_nums_dim);
  CHECK_VALID(roi_nums_tensor);
  auto pad_values_tensor = network->AddInput(int32_dt, pad_values_dim);
  CHECK_VALID(pad_values_tensor);
  // add pluginop
  magicmind::TensorMap plugin_inputs;
  std::vector<magicmind::ITensor *> input{input_tensor};
  std::vector<magicmind::ITensor *> crop_params{crop_params_tensor};
  std::vector<magicmind::ITensor *> roi_nums{roi_nums_tensor};
  std::vector<magicmind::ITensor *> pad_values{pad_values_tensor};
  plugin_inputs["input"] = input;
  plugin_inputs["crop_params"] = crop_params;
  plugin_inputs["roi_nums"] = roi_nums;
  plugin_inputs["pad_values"] = pad_values;
  magicmind::DataTypeMap plugin_outputs_dtype;
  plugin_outputs_dtype["output"] = {uint8_dt};

  magicmind::IPluginNode *plugin_crop_and_resize =
      network->AddIPluginNode("PluginCropAndResize", plugin_inputs, plugin_outputs_dtype);
  CHECK_VALID(plugin_crop_and_resize);
  CHECK_STATUS(plugin_crop_and_resize->SetAttr("d_row", d_row));
  CHECK_STATUS(plugin_crop_and_resize->SetAttr("d_col", d_col));
  CHECK_STATUS(plugin_crop_and_resize->SetAttr("keep_aspect_ratio", keep_aspect_ratio));

  // set outputs nodes
  CHECK_STATUS(network->MarkOutput(plugin_crop_and_resize->GetOutput(0)));

  // create model
  auto model = SUniquePtr<magicmind::IModel>(
      builder->BuildModel("plugin_crop_sample_model", network.get(), config.get()));
  CHECK_VALID(model);
  CHECK_STATUS(model->SerializeToFile(model_name));
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    SLOG(INFO) << "plugin_crop_sample_test usage: ";
    SLOG(INFO)
        << "  ./${path_to_executable_dir}/plugin_crop_demo ${mm_plugin_so_dir_path} ";
    return -1;
  }
  std::string lib_path = std::string(argv[1]);
  auto kernel_lib = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (!kernel_lib) {
    SLOG(ERROR) << "Call dlopen() failed : " << dlerror();
    abort();
  }

  std::string model_name = "plugin_crop_sample_model";
  magicmind::DataType uint8_dt = magicmind::DataType::UINT8;
  magicmind::DataType int32_dt = magicmind::DataType::INT32;
  int batch_size = 4;
  int channel = 4;
  int s_row = 1080;
  int s_col = 608;
  int total_rois = 1 + 2 + 3 + 4;
  int xywh = 4;

  int64_t d_row = 200;
  int64_t d_col = 200;
  int64_t keep_aspect_ratio = 1;

  auto input_dim = magicmind::Dims({batch_size, channel, s_row, s_col});
  auto crop_params_dim = magicmind::Dims({total_rois, xywh});
  auto roi_nums_dim = magicmind::Dims({batch_size});
  auto pad_values_dim = magicmind::Dims({xywh});

  ConstructCropAndResizeNetwork(uint8_dt, int32_dt, input_dim, crop_params_dim, roi_nums_dim,
                                pad_values_dim, d_col, d_row, keep_aspect_ratio,
                                model_name.c_str());
  dlclose(kernel_lib);
  return 0;
}
