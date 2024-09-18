/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <mutex>
#include <iostream>
#include "cnrt.h"
#include "common/device.h"
#include "common/logger.h"
#include "common/macros.h"
#include "basic_samples/sample_refit/worker.h"

static void FillDevice(void *dev_ptr, size_t size, float v) {
  std::vector<float> data(size / sizeof(float), v);
  CHECK_CNRT(cnrtMemcpy(dev_ptr, data.data(), size, CNRT_MEM_TRANS_DIR_HOST2DEV));
}

using namespace magicmind;

void Worker(IEngine *engine, int id, std::atomic<bool> *is_should_stop) {
  static std::mutex context_creation_mutex;
  SLOG(INFO) << "Worker " << id << " started.";
  IContext *context = nullptr;
  {
    // IEngine::CreateIContext does not guarantee to be thread-safe on same engine object.
    std::lock_guard<std::mutex> lock(context_creation_mutex);
    context = engine->CreateIContext();
    CHECK_VALID(context);
  }
  cnrtQueue_t queue;
  CHECK_CNRT(cnrtQueueCreate(&queue));

  std::vector<magicmind::IRTTensor *> input_tensors;
  std::vector<magicmind::IRTTensor *> output_tensors;
  CHECK_STATUS(context->CreateInputTensors(&input_tensors));
  CHECK_STATUS(context->CreateOutputTensors(&output_tensors));

  CHECK_STATUS(input_tensors[0]->SetDimensions(Dims({1, 224, 224, 3})));
  CHECK_STATUS(context->InferOutputShape(input_tensors, output_tensors));

  void *input_dev_ptr;
  CHECK_CNRT(cnrtMalloc(&input_dev_ptr, input_tensors[0]->GetSize()));
  CHECK_STATUS(input_tensors[0]->SetData(input_dev_ptr));

  void *out_put_dev_ptr;
  CHECK_CNRT(cnrtMalloc(&out_put_dev_ptr, output_tensors[0]->GetSize()));
  CHECK_STATUS(output_tensors[0]->SetData(out_put_dev_ptr));

  int output_id = 0;
  int output_size = output_tensors[0]->GetSize();
  std::vector<float> last_output(output_tensors[0]->GetSize() / sizeof(float), 0);
  std::vector<float> this_output(output_tensors[0]->GetSize() / sizeof(float), 0);
  while (!is_should_stop->load()) {
    FillDevice(input_dev_ptr, input_tensors[0]->GetSize(), 1.0);
    FillDevice(out_put_dev_ptr, output_tensors[0]->GetSize(), 0.0);

    CHECK_STATUS(context->Enqueue(input_tensors, output_tensors, queue));
    CHECK_CNRT(cnrtQueueSync(queue));

    CHECK_CNRT(
        cnrtMemcpy(this_output.data(), out_put_dev_ptr, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    if (last_output != this_output) {
      SLOG(INFO) << "Worker " << id << " get different output at " << output_id;
    }
    std::swap(last_output, this_output);
    ++output_id;
  }
  CHECK_CNRT(cnrtFree(input_dev_ptr));
  CHECK_CNRT(cnrtFree(out_put_dev_ptr));
  input_tensors[0]->Destroy();
  output_tensors[0]->Destroy();
  CHECK_CNRT(cnrtQueueDestroy(queue));
  context->Destroy();
  SLOG(INFO) << "Worker " << id << " stopped.";
}
