/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include <dlfcn.h>
#include "mm_builder.h"
#include "mm_network.h"
#include "common/logger.h"
#include "common/macros.h"
#include "common/param.h"
#include "common/container.h"
void ConstructSpatialTransformNetwork(magicmind::DataType input_datatype,
                                      magicmind::Dims input_dim,
                                      magicmind::Dims output_dim,
                                      magicmind::Dims mat_dim,
                                      magicmind::Dims multable_value_dim,
                                      const std::vector<magicmind::Layout> &layout_vec,
                                      const char *model_name) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto config = SUniquePtr<magicmind::IBuilderConfig>(magicmind::CreateIBuilderConfig());
  CHECK_VALID(config);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);
  // create input tensor
  auto input_tensor = network->AddInput(input_datatype, input_dim);
  CHECK_VALID(input_tensor);
  auto mat_tensor = network->AddInput(input_datatype, mat_dim);
  CHECK_VALID(mat_tensor);
  auto multable_value_tensor = network->AddInput(input_datatype, multable_value_dim);
  CHECK_VALID(multable_value_tensor);
  // add SpatialTransform pluginop
  magicmind::TensorMap plugin_inputs;
  std::vector<magicmind::ITensor *> input{input_tensor};
  std::vector<magicmind::ITensor *> mat{mat_tensor};
  std::vector<magicmind::ITensor *> multable_value{multable_value_tensor};
  plugin_inputs["input"] = input;
  plugin_inputs["mat"] = mat;
  plugin_inputs["multable_value"] = multable_value;
  magicmind::DataTypeMap plugin_outputs_dtype;
  plugin_outputs_dtype["output"] = {input_datatype};

  magicmind::IPluginNode *plugin_spatial_transform =
      network->AddIPluginNode("PluginSpatialTransform", plugin_inputs, plugin_outputs_dtype);
  CHECK_VALID(plugin_spatial_transform);
  CHECK_STATUS(plugin_spatial_transform->SetAttr("op_layout", layout_vec));

  // set outputs nodes
  CHECK_STATUS(network->MarkOutput(plugin_spatial_transform->GetOutput(0)));

  // creat model
  auto model =
      SUniquePtr<magicmind::IModel>(builder->BuildModel(model_name, network.get(), config.get()));
  CHECK_VALID(model);
  CHECK_STATUS(model->SerializeToFile(model_name));
}

class SpatialTransArg : public ArgListBase {
  DECLARE_ARG(plugin_lib, (std::string))->SetDescription("Plugin kernel library.");
  DECLARE_ARG(layout, (std::string))
      ->SetDescription("Input/Output layout")
      ->SetAlternative({"NCHW", "NHWC"});
  DECLARE_ARG(datatype, (std::string))
      ->SetDescription("Input/Output datatype")
      ->SetAlternative({"float", "half"});
};

int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  SpatialTransArg arg_reader;
  arg_reader.ReadIn(args);
  auto plugin_lib = Value(arg_reader.plugin_lib());
  auto layout = Value(arg_reader.layout());
  auto datatype = Value(arg_reader.datatype());
  auto kernel_lib = dlopen(plugin_lib.c_str(), RTLD_LAZY);
  if (!kernel_lib) {
    SLOG(ERROR) << "Call dlopen() failed : " << dlerror();
    abort();
  }
  std::string model_name = "spatial_transform_model_" + layout + "_" + datatype;
  magicmind::DataType input_datatype = magicmind::DataType::FLOAT32;
  if (datatype == "half") {
    input_datatype = magicmind::DataType::FLOAT16;
  }
  int64_t batch_size = 2;
  int64_t src_h = 40;
  int64_t src_w = 180;
  int64_t c = 1;

  magicmind::Dims input_dim;
  magicmind::Dims output_dim;
  std::vector<magicmind::Layout> layout_vec(4, magicmind::Layout::ARRAY);
  if (layout == "NHWC") {
    input_dim = magicmind::Dims({batch_size, src_h, src_w, c});
    output_dim = magicmind::Dims({batch_size, src_h, src_w, c});
    layout_vec[0] = magicmind::Layout::NHWC;
    layout_vec[3] = magicmind::Layout::NHWC;
  } else {
    input_dim = magicmind::Dims({batch_size, c, src_h, src_w});
    output_dim = magicmind::Dims({batch_size, c, src_h, src_w});
    layout_vec[0] = magicmind::Layout::NCHW;
    layout_vec[3] = magicmind::Layout::NCHW;
  }

  auto mat_dim = magicmind::Dims({batch_size, 6});
  auto multable_value_dim = magicmind::Dims({batch_size, 2});

  ConstructSpatialTransformNetwork(input_datatype, input_dim, output_dim, mat_dim,
                                   multable_value_dim, layout_vec, model_name.c_str());

  dlclose(kernel_lib);
  return 0;
}
