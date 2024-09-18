/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include "cnrt.h"
#include "common/device.h"
#include "common/logger.h"
#include "common/macros.h"
#include "common/container.h"
#include "basic_samples/sample_refit/refit_model.h"
#include "basic_samples/sample_refit/worker.h"

using namespace magicmind;

IRTTensor *CreateFixedValueHostIRTensor(float *data,
                                        const Dims &dims,
                                        magicmind::Layout layout = magicmind::Layout::ARRAY) {
  auto *ret = CreateIRTTensor(DataType::FLOAT32, "_unused", layout, TensorLocation::kHost);
  CHECK_VALID(ret);
  CHECK_STATUS(ret->SetDimensions(dims));
  CHECK_STATUS(ret->SetData(data));
  return ret;
}

void RunEngine(IModel *model, int engine_id, int worker_num, int refit_delay_sec) {
  static std::mutex engine_creation_mutex;
  IEngine *engine = nullptr;
  {
    // IModel::CreateIEngine does not guarantee to be thread-safe on same engine object.
    std::lock_guard<std::mutex> lock(engine_creation_mutex);
    engine = model->CreateIEngine();
    CHECK_VALID(engine);
  }
  // Start workers
  std::vector<std::atomic<bool>> stop_signal(worker_num);
  std::vector<std::thread> workers(worker_num);

  for (int work_offset = 0; work_offset < worker_num; ++work_offset) {
    int worker_id = engine_id * worker_num + work_offset;
    stop_signal[work_offset].store(false);
    workers[work_offset] = std::thread(Worker, engine, worker_id, &stop_signal[work_offset]);
  }
  // Sleep a while to let workers work
  std::this_thread::sleep_for(std::chrono::seconds(refit_delay_sec));

  // Update Engine using refitter
  IRefitter *refitter = CreateIRefitter(engine);
  CHECK_VALID(refitter);
  // Create a message to log
  std::vector<std::string> missing_weight_names;
  CHECK_STATUS(refitter->GetMissingNames(&missing_weight_names));
  std::string msg = "Engine " + std::to_string(engine_id) + " weights updating: ";
  for (auto &name : missing_weight_names) {
    msg += " ";
    msg += name;
  }

  // Set all with new value 2.0f
  std::vector<IRTTensor *> to_destroy;
  std::vector<float> new_filter_data(32 * 3 * 3 * 3, 2);
  auto *new_filter =
      CreateFixedValueHostIRTensor(new_filter_data.data(), Dims({32, 3, 3, 3}), Layout::NHWC);
  to_destroy.push_back(new_filter);
  CHECK_STATUS(refitter->SetNamedWeights("conv0_filter", new_filter));

  std::vector<float> new_bias_data(32, 2);
  auto *new_bias = CreateFixedValueHostIRTensor(new_bias_data.data(), Dims({32}));
  to_destroy.push_back(new_bias);
  CHECK_STATUS(refitter->SetNamedWeights("conv0_bias", new_bias));

  std::vector<float> new_prod_b_data(224 * 32 * 10, 2);
  auto *new_prod_b = CreateFixedValueHostIRTensor(new_prod_b_data.data(), Dims({224 * 32, 10}));
  to_destroy.push_back(new_prod_b);
  CHECK_STATUS(refitter->SetNamedWeights("prod_b_tensor", new_prod_b));

  std::vector<float> new_prod_bias_data(10, 2);
  auto *new_prod_bias = CreateFixedValueHostIRTensor(new_prod_bias_data.data(), Dims({10}));
  to_destroy.push_back(new_prod_bias);
  CHECK_STATUS(refitter->SetNamedWeights("prod_bias_tensor", new_prod_bias));

  missing_weight_names.clear();
  CHECK_STATUS(refitter->GetMissingNames(&missing_weight_names));
  CHECK_VALID(missing_weight_names.size() == 0);

  CHECK_STATUS(refitter->RefitEngine());
  SLOG(INFO) << msg;

  // Sleep a while to let workers work with updated weight
  std::this_thread::sleep_for(std::chrono::seconds(refit_delay_sec));

  // kill all workers and exit
  for (int work_offset = 0; work_offset < worker_num; ++work_offset) {
    stop_signal[work_offset].store(true);
    workers[work_offset].join();
  }
  for (auto tensor : to_destroy) {
    tensor->Destroy();
  }
  refitter->Destroy();
  CHECK_STATUS(engine->Destroy());
};

int main() {
  // Refit must be enabled in config when building model
  auto config = SUniquePtr<magicmind::IBuilderConfig>(magicmind::CreateIBuilderConfig());
  CHECK_VALID(config);
  CHECK_STATUS(config->ParseFromString(R"({"enable_refit": true})"));

  // Any model will be ok, here we just create a simple conv model
  auto model = CreateModel(config.get());

  // We will Create ENGINE_NUM engines, and every engine will create WORKER_NUM_PER_ENGINE workers,
  // every worker has their own context

  const int ENGINE_NUM = 2;
  const int WORKER_NUM_PER_ENGINE = 2;
  const int REFIT_DELAY = 5;

  CHECK_CNRT(cnrtSetDevice(0));

  std::vector<std::thread> engine_runners(ENGINE_NUM);
  for (int engine_id = 0; engine_id < ENGINE_NUM; ++engine_id) {
    engine_runners[engine_id] =
        std::thread(RunEngine, model, engine_id, WORKER_NUM_PER_ENGINE, REFIT_DELAY);
  }
  for (int i = 0; i < ENGINE_NUM; ++i) {
    engine_runners[i].join();
  }
  model->Destroy();
  return 0;
}
