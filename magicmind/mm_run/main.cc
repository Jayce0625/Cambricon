/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include <unistd.h>
#include "mm_run/run.h"
#include "mm_run/run_param.h"
int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  // arg parse
  auto param = new RunParam();
  param->ReadIn(args);
  SLOG(INFO) << "\n==================== Parameter Information\n"
             << param->DebugString() << "MagicMind: " << MM_VERSION_STR << std::endl
             << "CNRT: " << CNRT_VERSION << std::endl
             << "CNAPI: " << CN_VERSION << std::endl
             << "PID: " << getpid();
  // Init magicmind_run
  Run *run = new Run(param);
  // Start multi-thread inference
  run->RunInMultiDevices();
  delete run;
  delete param;
  return 0;
}
