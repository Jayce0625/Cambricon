/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#ifndef MAIN_PROCESS_H_
#define MAIN_PROCESS_H_
#include <vector>
#include <string>
#include "mm_builder.h"
#include "mm_calibrator.h"
#include "mm_builder.h"
#include "common/calib_data.h"
#include "mm_build/build_param.h"
#include "mm_build/parser.h"

// Wrapper for build config
IBuilderConfig *GetConfig(BuildParam *param);

bool ConfigNetwork(INetwork *net, BuildParam *param);

bool Calibration(INetwork *net, IBuilderConfig *config, BuildParam *param);

bool BuildAndSerialize(INetwork *net, IBuilderConfig *config, BuildParam *param);

template <ModelKind Kind>
int MainProcess(ParserParam<Kind> *param) {
  ModelParser<Kind> parser(param);
  auto net = CreateINetwork();
  CHECK_VALID(net);
  parser.Parse(net);
  CHECK_VALID(ConfigNetwork(net, param));
  auto config = GetConfig(param);
  CHECK_VALID(config);
  if (HasValue(param->calibration()) && Value(param->calibration())) {
    CHECK_VALID(Calibration(net, config, param));
  }
  CHECK_VALID(BuildAndSerialize(net, config, param));
  config->Destroy();
  net->Destroy();
  return 0;
}

#endif  // MAIN_PROCESS_H_
