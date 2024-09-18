/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <memory>
#include "common/calib_data.h"
#include "common/logger.h"
#include "common/macros.h"
#include "common/container.h"
#include "basic_samples/sample_calibration/sample_calibration.h"
#define CALIB_NUM 10
int SampleCalibration::ConstructNetwork(const std::string &prototxt_path,
                                        const std::string &caffemodel_path,
                                        const std::string &data_path,
                                        const std::string &rpc_server,
                                        const std::string &model_name) {
  // init builder, network, builder_config and parser
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);
  auto config = SUniquePtr<magicmind::IBuilderConfig>(magicmind::CreateIBuilderConfig());
  CHECK_VALID(config);
  auto parser =
      SUniquePtr<magicmind::IParser<magicmind::ModelKind::kCaffe, std::string, std::string>>(
          magicmind::CreateIParser<magicmind::ModelKind::kCaffe, std::string, std::string>());
  CHECK_VALID(parser);

  // 1. Parse the caffe model using the parser
  CHECK_STATUS(config->ParseFromString(
      R"({"precision_config": {"precision_mode": "qint8_mixed_float32"}})"));
  CHECK_STATUS(parser->Parse(network.get(), caffemodel_path, prototxt_path));
  magicmind::Dims input_dims = network->GetInput(0)->GetDimension();

  // get calibration data
  std::vector<std::string> data_paths;
  std::string data_path_prefix = data_path + "/";
  int count = 0;
  for (auto iter = images_.begin(); iter != images_.end() && count < CALIB_NUM; iter++) {
    data_paths.push_back(data_path_prefix + *iter);
    count++;
  }

  if (input_dims.GetElementCount() == -1) {
    SLOG(ERROR)
        << "Can not get elmentcount of calibration set, maybe there is one -1 in its shape.";
    return 1;
  }
  // 2. Initialize calibrator with CalibDataInterface and then do calibration.
  // 2.1. Create CalibData object
  SampleCalibData calib_data(input_dims, magicmind::DataType::FLOAT32, data_paths.size(),
                             data_paths);
  auto algorithm = magicmind::QuantizationAlgorithm::LINEAR_ALGORITHM;
  SUniquePtr<magicmind::ICalibrator> calibrator(magicmind::CreateICalibrator(&calib_data));
  CHECK_VALID(calibrator);
  if (rpc_server.size()) {
    magicmind::RemoteConfig remote_config;
    remote_config.address = rpc_server;
    remote_config.device_id = 0;
    CHECK_STATUS(calibrator->SetRemote(remote_config));
  }
  CHECK_STATUS(calibrator->SetQuantizationAlgorithm(algorithm));
  CHECK_STATUS(
      config->ParseFromString(R"({"precision_config": {"weight_quant_granularity": "per_axis"}})"));
  CHECK_STATUS(calibrator->Calibrate(network.get(), config.get()));

  // 3. Build model with network and config.
  auto quantized_model =
      SUniquePtr<magicmind::IModel>(builder->BuildModel(model_name, network.get(), config.get()));
  CHECK_VALID(quantized_model);

  // 4. Serialize quantized model.
  CHECK_STATUS(quantized_model->SerializeToFile(model_name.c_str()));
  return 0;
}

int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  CalibArg arg_reader;
  arg_reader.ReadIn(args);
  auto data_path = arg_reader.data_path();
  auto label_path = arg_reader.label_path();
  auto prototxt_path = arg_reader.prototxt_path();
  auto caffemodel_path = arg_reader.caffemodel_path();
  auto rpc_server = arg_reader.rpc_server();
  std::string model_name = "model_quant";
  SampleCalibration cali(Value(label_path));
  return cali.ConstructNetwork(Value(prototxt_path), Value(caffemodel_path), Value(data_path), Value(rpc_server),
                               model_name);
}
