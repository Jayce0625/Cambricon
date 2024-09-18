/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "mm_build/main_process.h"
#include "mm_build/build_param.h"

template <ModelKind Kind>
int call_main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  auto param = new ParserParam<Kind>();
  param->ReadIn(args);
  SLOG(INFO) << "\n==================== Parameter Information\n"
             << param->DebugString() << "MagicMind: " << MM_VERSION_STR;
  auto ret = MainProcess<Kind>(param);
  delete param;
  return ret;
}

#ifdef BUILD_CAFFE
int main(int argc, char *argv[]) {
  return call_main<ModelKind::kCaffe>(argc, argv);
}
#endif

#ifdef BUILD_TENSORFLOW
int main(int argc, char *argv[]) {
  return call_main<ModelKind::kTensorflow>(argc, argv);
}
#endif

#ifdef BUILD_PYTORCH
int main(int argc, char *argv[]) {
  return call_main<ModelKind::kPytorch>(argc, argv);
}
#endif

#ifdef BUILD_ONNX
int main(int argc, char *argv[]) {
  return call_main<ModelKind::kOnnx>(argc, argv);
}
#endif
