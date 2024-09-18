/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLES_BASIC_SAMPLES_SAMPLE_CALIBRATION_SAMPLE_CALIBRATION_H_
#define SAMPLES_BASIC_SAMPLES_SAMPLE_CALIBRATION_SAMPLE_CALIBRATION_H_
#include <vector>
#include <string>
#include "mm_builder.h"
#include "mm_calibrator.h"
#include "mm_parser.h"
#include "common/data.h"
#include "common/param.h"
class SampleCalibration {
 public:
  explicit SampleCalibration(const std::string &label_txt) {
    CHECK_VALID(ReadLabelFromFile(label_txt, &images_, &labels_));
  }
  int ConstructNetwork(const std::string &prototxt_path,
                       const std::string &caffemodel_path,
                       const std::string &data_path,
                       const std::string &rpc_server,
                       const std::string &model_name);

 private:
  std::vector<std::string> images_;
  std::vector<int> labels_;
};

class CalibArg : public ArgListBase {
  DECLARE_ARG(data_path, (std::string))->SetDescription("Data path for sample calibration.");
  DECLARE_ARG(label_path, (std::string))->SetDescription("Img label path for sample calibration.");
  DECLARE_ARG(prototxt_path, (std::string))
      ->SetDescription("Caffe prototxt file path for sample calibration.");
  DECLARE_ARG(caffemodel_path, (std::string))
      ->SetDescription("Caffe .caffemodel file path for sample calibration.");
  DECLARE_ARG(rpc_server, (std::string))
      ->SetDescription("Set remote address for calibration.")
      ->SetDefault({});
};

#endif  //  SAMPLES_BASIC_SAMPLES_SAMPLE_CALIBRATION_SAMPLE_CALIBRATION_H_
