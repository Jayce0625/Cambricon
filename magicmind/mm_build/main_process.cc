/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include <dlfcn.h>
#include "common/logger.h"
#include "common/macros.h"
#include "common/type.h"
#include "mm_build/main_process.h"

namespace {
bool ConvertRGB2BGRForFirstLayer(INetwork *net) {
  std::vector<INode *> nodes = net->FindNodesByNodeType("IConvNode");
  if (nodes.size() == 0) {
    SLOG(INFO) << "Not a cv network, so no need to convert rgb to bgr format.";
    return false;
  }
  ITensor *first_weight_node = nodes[0]->GetInput(1);
  std::vector<int64_t> dims = first_weight_node->GetDimension().GetDims();
  IConvNode *conv_node = static_cast<IConvNode *>(nodes[0]);
  CHECK_VALID(conv_node);
  Layout input_layout;
  Layout weight_layout;
  Layout output_layout;
  CHECK_STATUS(conv_node->GetLayout(&input_layout, &weight_layout, &output_layout));
  int channel_num = 0;
  int axis = -1;
  if (weight_layout == magicmind::Layout::NCHW) {
    channel_num = dims[1];
    axis = 1;
  } else if (weight_layout == magicmind::Layout::NHWC) {
    channel_num = dims[3];
    axis = 3;
  } else if (weight_layout == magicmind::Layout::HWCN) {
    channel_num = dims[2];
    axis = 2;
  } else {
    SLOG(INFO) << "Invalid layout: " << weight_layout;
    return false;
  }
  if (channel_num != 3) {
    SLOG(INFO) << "Channel's num of Conv weight is not 3.";
    return false;
  }

  std::vector<INode *> precursors = nodes[0]->GetPrecursors();
  std::pair<INode *, ITensor *> node_weight_pair;
  bool found_scale_or_batch_norm = false;
  for (size_t i = 0; i < precursors.size(); ++i) {
    if (precursors[i]->GetNodeType() == "IScaleNode" ||
        precursors[i]->GetNodeType() == "IFusedBatchNormNode") {
      node_weight_pair = std::make_pair(precursors[i], precursors[i]->GetInput(0));
      found_scale_or_batch_norm = true;
      break;
    }
  }
  if (!found_scale_or_batch_norm) {
    node_weight_pair = std::make_pair(nodes[0], nodes[0]->GetInput(1));
  }
  // convert rgb to bgr for weight_tensor, that is node_weight_pair.second
  std::vector<int> s{axis};
  auto const_split = net->AddIConstNode(magicmind::DataType::INT32, magicmind::Dims({1}), s.data());
  auto split_node = net->AddISplitNode(node_weight_pair.second, const_split->GetOutput(0), 3);
  std::vector<magicmind::ITensor *> concat_in{split_node->GetOutput(2), split_node->GetOutput(1),
                                              split_node->GetOutput(0)};
  auto concat_node = net->AddIConcatNode(const_split->GetOutput(0), concat_in);
  auto need_to_convert_node = node_weight_pair.first;
  if (found_scale_or_batch_norm) {
    CHECK_STATUS(need_to_convert_node->UpdateInput(0, concat_node->GetOutput(0)));
  } else {
    CHECK_STATUS(need_to_convert_node->UpdateInput(1, concat_node->GetOutput(0)));
  }
  return true;
}
std::string GetChannelOppositeLayout(const std::string &in) {
  static const std::unordered_map<std::string, std::string> layouts = {
      {"NCT", "NTC"}, {"NCHW", "NHWC"}, {"NCDHW", "NDHWC"}};
  for (auto e_ : layouts) {
    if (in == e_.first) {
      return e_.second;
    } else if (in == e_.second) {
      return e_.first;
    }
  }
  SLOG(ERROR) << "Unsupport layout convertion";
  abort();
}

}  // namespace

void BindCluster(std::stringstream *ss,
                 const std::vector<std::string> &mlu_arch,
                 const std::vector<std::vector<int>> &cluster_num,
                 size_t index) {
  *ss << "{\"";
  *ss << mlu_arch[index];
  *ss << "\":";
  if (cluster_num.size() > index) {
    *ss << "[";
    for (size_t j = 0; j < cluster_num[index].size(); ++j) {
      *ss << std::to_string(cluster_num[index][j]);
      if (j < cluster_num[index].size() - 1) {
        *ss << ",";
      }
    }
    *ss << "]}";
  } else {
    // support empty array
    *ss << "[]}";
  }
  return;
} 
IBuilderConfig *GetConfig(BuildParam *param) {
  auto config_ptr_ = CreateIBuilderConfig();
  CHECK_VALID(config_ptr_);
  // arch
  if (!Value(param->mlu_arch()).empty()) {
    CHECK_STATUS(config_ptr_->SetMLUArch(Value(param->mlu_arch())));
  }

  // precision
  if (HasValue(param->precision())) {
    CHECK_STATUS(config_ptr_->ParseFromString("{\"precision_config\":{\"precision_mode\":\"" +
                                              Value(param->precision()) + "\"}}"));
  }
  // cluster num
  auto cluster_num =  Value(param->cluster_num());
  auto mlu_arch =  Value(param->mlu_arch());
  if (cluster_num.size() > 0 && mlu_arch.size() > 0) {
    // set bitmap of visible cluster for each architecture
    // example: "archs": [{"mtp_372": [cluster_num_1, cluster_num_2, cluster_num_3]}]
    std::stringstream ss;
    ss << "{\"archs\": [";
    for (size_t i = 0; i < mlu_arch.size(); ++i) {
      BindCluster(&ss, mlu_arch, cluster_num, i);
      if (i < mlu_arch.size() - 1) {
        ss << ",";
      } else {
        ss << "]";
      }
    }
    ss << "}";
    CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
  }

  // means/vars
  if (HasValue(param->means())) {
    CHECK_VALID(HasValue(param->vars()));
    auto means = Value(param->means());
    auto vars = Value(param->vars());
    std::stringstream ss;
    ss << "{\"insert_bn_before_firstnode\":{";
    for (size_t i = 0; i < means.size(); ++i) {
      ss << "\"" << std::to_string(i) << "\":{\"mean\":";
      ss << means[i];
      ss << ",\"var\":";
      ss << vars[i];
      ss << "}";
      if (i < means.size() - 1) {
        ss << ",";
      } else {
        ss << "}";
      }
    }
    ss << "}";
    CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
  }
  // cross_compile
  if (HasValue(param->toolchain_path())) {
    CHECK_STATUS(config_ptr_->ParseFromString("{\"cross_compile_toolchain_path\":\"" +
                                              Value(param->toolchain_path()) + "\"}"));
  }
  // layout in/out
  if (HasValue(param->input_layout())) {
    std::stringstream ss;
    auto v = Value(param->input_layout());
    ss << "{\"convert_input_layout\":{";
    for (size_t i = 0; i < v.size(); ++i) {
      ss << "\"" << std::to_string(i) << "\":{\"src\":\"";
      ss << GetChannelOppositeLayout(v[i]) << "\",";
      ss << "\"dst\":\"";
      ss << v[i];
      ss << "\"}";
      if (i < v.size() - 1) {
        ss << ",";
      } else {
        ss << "}";
      }
    }
    ss << "}";
    CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
  }
  if (HasValue(param->output_layout())) {
    std::stringstream ss;
    auto v = Value(param->output_layout());
    ss << "{\"convert_output_layout\":{";
    for (size_t i = 0; i < v.size(); ++i) {
      ss << "\"" << std::to_string(i) << "\":{\"src\":\"";
      ss << GetChannelOppositeLayout(v[i]) << "\",";
      ss << "\"dst\":\"";
      ss << v[i];
      ss << "\"}";
      if (i < v.size() - 1) {
        ss << ",";
      } else {
        ss << "}";
      }
    }
    ss << "}";
    CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
  }
  if (HasValue(param->dynamic_shape())) {
    if (!Value(param->dynamic_shape())) {
      CHECK_STATUS(config_ptr_->ParseFromString("{\"graph_shape_mutable\":false}"));
    }
  }
  // rest json
  if (HasValue(param->build_config())) {
    CHECK_STATUS(config_ptr_->ParseFromFile(Value(param->build_config())));
  }
  return config_ptr_;
}

bool ConfigNetwork(INetwork *net, BuildParam *param) {
  size_t input_count = net->GetInputCount();
  if (HasValue(param->input_dims())) {
    SLOG(INFO) << "Reset input dims.";
    auto dims = ToDims(Value(param->input_dims()));
    if (input_count != dims.size()) {
      SLOG(ERROR) << "Got " << input_count << " inputs from network, but " << dims.size()
                  << " from param->";
      return false;
    }
    for (size_t i = 0; i < input_count; ++i) {
      auto tensor = net->GetInput(i);
      if (!tensor->SetDimension(dims[i]).ok()) {
        return false;
      }
    }
  }
  if (HasValue(param->batch_size())) {
    SLOG(INFO) << "Reset input batch size (highest dimension)";
    auto batch_sizes = Value(param->batch_size());
    for (size_t i = 0; i < input_count; ++i) {
      auto tensor = net->GetInput(i);
      auto dim = tensor->GetDimension().GetDims();
      if (dim.size() > 0) {
        dim[0] = batch_sizes[i];
        if (!tensor->SetDimension(magicmind::Dims(dim)).ok()) {
          return false;
        }
      } else {
        SLOG(INFO) << "Input " << i << " rank is " << dim.size()
                   << ", batch_size param will not be applied.";
      }
    }
  }
  /////////////Other user action below////////////////
  if (HasValue(param->rgb2bgr()) && Value(param->rgb2bgr())) {
    SLOG(INFO) << "Convert RGB to BGR for first layer's Conv/BatchNorm/Scale of network";
    CHECK_VALID(ConvertRGB2BGRForFirstLayer(net));
  }
  /////////////Other user action end here/////////////
  return true;
}

bool Calibration(INetwork *net, IBuilderConfig *config, BuildParam *param) {
  bool rand_calib = false;
  bool fake_calib = false;
  std::vector<float> value;
  if (HasValue(param->random_calib_range())) {
    value = Value(param->random_calib_range());
    if (value.size() != 2) {
      SLOG(ERROR) << "Random range for calibration must be two numbers (min, max).";
      return false;
    }
    SLOG(INFO) << "Do rand calibration with custom range: " << value;
    rand_calib = true;
  } else if (!HasValue(param->file_list()) && !HasValue(param->calibration_data_path())) {
    SLOG(INFO) << "Fill fixed quant param without real calibration.";
    fake_calib = true;
  } else if (!HasValue(param->file_list()) || !HasValue(param->calibration_data_path())) {
    SLOG(ERROR) << "Calibration list file and path must be provided together.";
    return false;
  }
  if (fake_calib) {
    CHECK_STATUS(config->ParseFromString(R"({"custom_ranges": {"" : {"max": [1], "min": [-1]}}})"));
  }
  std::vector<CalibDataInterface *> calib_datas(net->GetInputCount());
  auto file_lists = Value(param->file_list());
  auto calibration_data_path = Value(param->calibration_data_path());
  size_t input_num = net->GetInputCount();
  if (!rand_calib && !fake_calib && (file_lists.size() != input_num)) {
    SLOG(ERROR) << "Got " << input_num << " inputs from network, but " << file_lists.size()
                << " from calibration file list.";
    return false;
  }
  for (size_t i = 0; i < input_num; ++i) {
    auto data_type = net->GetInput(i)->GetDataType();
    auto input_dim = net->GetInput(i)->GetDimension();
    if (fake_calib) {
      // Init empty calibration data. Calibration process will be passed in fake calibration.
      calib_datas[i] = new SampleCalibData(input_dim, data_type);
    } else if (rand_calib) {
      // Init rand calibration data.
      calib_datas[i] =
          new SampleCalibData(input_dim, data_type, input_dim.GetDimValue(0), value[0], value[1]);
    } else {
      // Init calibration data from files.
      std::vector<std::string> files;
      std::vector<std::vector<int>> shapes;
      if (!ReadListFromFile(file_lists[i], &files, &shapes)) {
        SLOG(ERROR) << "Read file list failed.";
        return false;
      }
      for (size_t f = 0; f < files.size(); ++f) {
        files[f] = calibration_data_path + "/" + files[f];
      }
      if (shapes.size()) {
        // With shape size in file list
        auto dims = ToDims(shapes);
        calib_datas[i] = new SampleCalibData(dims, data_type, files.size(), files);
      } else {
        // Use network shape as input shape.
        if (input_dim.GetElementCount() == -1) {
          SLOG(ERROR)
              << "Can not get elmentcount of calibration set, maybe there is one -1 in its shape.";
          return false;
        }
        calib_datas[i] = new SampleCalibData(input_dim, data_type, files.size(), files);
      }
    }
  }
  auto calibrator = CreateICalibrator(calib_datas);
  if (!calibrator) {
    SLOG(ERROR) << "Create calibrator failed.";
    return false;
  }
  if (HasValue(param->rpc_server())) {
    RemoteConfig remote_config;
    remote_config.address = Value(param->rpc_server());
    remote_config.device_id = 0;
    CHECK_STATUS(calibrator->SetRemote(remote_config));
  }
  CHECK_STATUS(calibrator->SetQuantizationAlgorithm(StringToAlgo(Value(param->calibration_algo()))));
  auto ret = calibrator->Calibrate(net, config);
  if (!ret.ok()) {
    SLOG(ERROR) << ret.ToString();
    return false;
  }
  for (auto &calib_data : calib_datas) {
    delete calib_data;
  }
  calibrator->Destroy();
  return true;
}

bool BuildAndSerialize(INetwork *net, IBuilderConfig *config, BuildParam *param) {
  auto builder = CreateIBuilder();
  auto model_name = Value(param->magicmind_model());
  if (!builder) {
    SLOG(ERROR) << "CreateIBuilder failed.";
    return false;
  }
  size_t input_count = net->GetInputCount();
  size_t output_count = net->GetOutputCount();
  if (HasValue(param->input_dtypes())) {
    SLOG(INFO) << "Reset input dtypes.";
    auto types = ToDataType(Value(param->input_dtypes()));
    if (input_count != types.size()) {
      SLOG(ERROR) << "Got " << input_count << " inputs from network, but " << types.size()
                  << " from param->";
      return false;
    }
    for (size_t i = 0; i < input_count; ++i) {
      auto tensor = net->GetInput(i);
      if (!tensor->SetDataType(types[i]).ok()) {
        return false;
      }
    }
  }
  if (HasValue(param->output_dtypes())) {
    SLOG(INFO) << "Reset output dtypes.";
    auto types = ToDataType(Value(param->output_dtypes()));
    if (output_count != types.size()) {
      SLOG(ERROR) << "Got " << output_count << " inputs from network, but " << types.size()
                  << " from param->";
      return false;
    }
    for (size_t i = 0; i < output_count; ++i) {
      auto tensor = net->GetOutput(i);
      if (!tensor->SetDataType(types[i]).ok()) {
        return false;
      }
    }
  }

  auto model = builder->BuildModel("network", net, config);
  if (!model) {
    SLOG(ERROR) << "BuildModel failed";
    return false;
  }
  auto ret = model->SerializeToFile(model_name.c_str());
  if (!ret.ok()) {
    SLOG(ERROR) << "Serialization failed with " << ret.ToString();
    return false;
  }
  auto input_dtypes = model->GetInputDataTypes();
  auto output_dtypes = model->GetOutputDataTypes();
  auto input_names = model->GetInputNames();
  auto output_names = model->GetOutputNames();
  for (int i = 0; i < model->GetInputNum(); i++) {
    auto input_shape = model->GetInputDimensions()[i];
    SLOG(INFO) << " model input[" << i << "] name is : " << input_names[i];
    SLOG(INFO) << " model input[" << i << "] shape is : " << input_shape;
    SLOG(INFO) << " model input[" << i
               << "] dtype is : " << magicmind::TypeEnumToString(input_dtypes[i]);
  }
  for (int i = 0; i < model->GetOutputNum(); i++) {
    auto output_shape = model->GetOutputDimensions()[i];
    SLOG(INFO) << " model output[" << i << "] name is : " << output_names[i];
    SLOG(INFO) << " model output[" << i << "] shape is : " << output_shape;
    SLOG(INFO) << " model output[" << i
               << "] dtype is : " << magicmind::TypeEnumToString(output_dtypes[i]);
  }
  builder->Destroy();
  model->Destroy();
  return true;
}
