/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLE_RUNTIME_H_
#define SAMPLE_RUNTIME_H_
#include <vector>
#include <string>
#include <thread>
#include <functional>
#include "mm_profiler.h"
#include "mm_remote.h"
#include "mm_runtime.h"
#include "common/param.h"
#include "common/type.h"
/*
 * An allocator class for cnrtMalloc.
 * In advance usages, users can make own memory pool and do memory
 * re-use by assign derived allocator to IEngine and IContext
 */
class PointerAdapterAllocator : public magicmind::IAllocator {
 public:
  PointerAdapterAllocator() {}

  void *AllocateRaw(uint64_t size, uint64_t alignment) override {
    uint64_t real_size = std::floor((size + alignment - 1) / alignment) * alignment;
    void *ptr          = nullptr;
    CHECK_CNRT(cnrtMalloc((void **)&ptr, real_size));
    return ptr;
  }
  void DeallocateRaw(void *ptr) override { CHECK_CNRT(cnrtFree(ptr)); }
};

// Objects for memory
enum MemoryStrategy {
  // Do nothing.
  kDefault = 0,
  // User allocate static memory and all intermediate result memory statically
  // by giving a fixed addr
  kUserStaticWorkSpace = 1,
  // User allocate static memory and all intermediate result memory dynamically
  // by giving an allocator
  kUserDynamicWorkSpace = 2,
};

/*
* Create and execute a IContext instance to complete model inference.
* Each IContext can bind to a specific cluster.
*/
class InferInstance {
 public:
  struct SetUp {
    magicmind::IEngine *engine = nullptr;
    std::vector<magicmind::Dims> in_dims;
    std::vector<std::string> data_path;
    std::vector<int64_t> output_capacities;
    std::vector<int> visible_cluster;
    std::string remote_server        = "";
    magicmind::IRpcSession *rpc_sess = nullptr;
    bool use_capacity                = false;
    bool bind_cluster                = false;
    bool use_random                  = false;
    bool do_dump                     = false;
    bool do_profile                  = false;
    MemoryStrategy mem_stg           = kDefault;
    int device_id                    = 0;
    int thread_id                    = 0;
  };
  InferInstance(const SetUp &set);
  ~InferInstance();

 private:
  void MallocIO();
  // Create context with different memory use strategy in local/remote.
  void CreateIContextAndSetInferWorkSpace();
  // Malloc inputs/outputs for infer.
  void Memcpy(void *dst, void *src, size_t size, bool host_to_mlu);
  // Do enqueue in given device.
  void EnqueueAndDumpAndProfile();
  void CopyOut();

 private:
  SetUp set_;
  // MagicMind vars
  magicmind::IContext *context_    = nullptr;
  void *intermedia_workspace_addr_ = nullptr;
  std::vector<magicmind::IRTTensor *> input_tensors_;
  std::vector<magicmind::IRTTensor *> output_tensors_;
  std::vector<void *> input_host_ptrs_;
  std::vector<void *> output_host_ptrs_;
  bool dynamic_output_             = false;
  magicmind::IProfiler *profiler_  = nullptr;
  cnrtQueue_t queue_               = nullptr;
  magicmind::IRpcDevice *dev_      = nullptr;
  magicmind::IRpcQueue *queue_rpc_ = nullptr;
};

/*
 * A straight-forward runtime sample for infer model, with prepared input shapes,
 * and no dynamic output shapes in given model.
 * For more advanced usages, e.g.,
 * encapsulations/double buffer/pipelines/dynamic inputs&&outputs,
 * see samples/cc/common/device.h, samples/cc/common/buffer.h
 * and samples/cc/mm_run.
 */
class RuntimeSample {
 public:
  RuntimeSample(const std::string &model_name,
                const std::vector<std::vector<int>> &input_shapes,
                const std::vector<std::string> &input_data,
                const std::vector<int64_t> &output_capacities,
                const std::string &rpc,
                int memory_strategy,
                int threads,
                bool do_profile,
                bool do_dump,
                std::vector<std::vector<int>> visible_cluster,
                int dev_id,
                const std::vector<std::string> &plugin_libs);
  void DoInfer();

 private:
  // Stages for infer model in runtime, each function states one part of key features
  // in runtime
  // Prepare plugin kernels to run (local only).
  void PreparePlugin();
  // Create model and print out its basic informations.
  void CreateIModelAndQueryInfo();
  // Create engine with different memory use strategy in local/remote.
  void CreateIEngineAndSetConstWorkSpace();
  // Clear eveything.
  void Destroy();

 private:
  // Common objects for inference
  std::string model_name_;
  int dev_id_ = 0;
  int threads_;
  std::vector<magicmind::Dims> in_dims_;
  magicmind::IModel *model_   = nullptr;
  magicmind::IEngine *engine_ = nullptr;
  // Objects for memory
  MemoryStrategy mem_stg_ = MemoryStrategy::kDefault;
  // intermedia and consts
  PointerAdapterAllocator *allocator_ = nullptr;
  void *const_workspace_addr_         = nullptr;
  // Objects for rpc
  std::string remote_server_    = "";
  magicmind::IRpcSession *sess_ = nullptr;
  // Objects for profile
  magicmind::IProfiler *profiler_   = nullptr;
  std::vector<std::string> plugins_ = {};
  std::vector<void *> plugin_libs_  = {};
  InferInstance::SetUp setup_;
  std::vector<std::thread> thds_;
  std::vector<std::vector<int>> visible_cluster_;
};

class InferArg : public ArgListBase {
  DECLARE_ARG(model_path, (std::string))
      ->SetDescription("Cambricon model file path for sample runtime.");
  DECLARE_ARG(data_path, (std::vector<std::string>))
      ->SetDescription("Data path for sample runtime.")
      ->SetDefault({});
  DECLARE_ARG(dev_ids, (std::vector<int>))
      ->SetDescription("Device id for sample runtime.")
      ->SetDefault({"0"});
  DECLARE_ARG(input_dims, (std::vector<std::vector<int>>))
      ->SetDescription("Input shapes for sample runtime.");
  DECLARE_ARG(rpc_server, (std::string))->SetDescription("Remote server address")->SetDefault({});
  DECLARE_ARG(mem_stg, (std::string))
      ->SetDescription("User MemoryStrategy.")
      ->SetAlternative({"static", "dynamic"})
      ->SetDefault({});
  DECLARE_ARG(profile, (bool))->SetDescription("To do profile during infer")->SetDefault({"false"});
  DECLARE_ARG(dump, (bool))->SetDescription("To do dump during infer")->SetDefault({"false"});
  DECLARE_ARG(plugin_libs, (std::vector<std::string>))
      ->SetDescription("Plugin kernel librarys")
      ->SetDefault({});
  DECLARE_ARG(threads, (int))
      ->SetDescription(
          "Thread num for launch jobs, each thread will run a context with separate cluster.")
      ->SetDefault({"1"});
  DECLARE_ARG(output_caps, (std::vector<int64_t>))
      ->SetDescription("Output capacities for sample runtime.")
      ->SetDefault({});
  DECLARE_ARG(visible_cluster, (std::vector<std::vector<int>>))
      ->SetDescription(
          "Values of visible cluster bitmap, each context will bind on one or more certain clusters.")
      ->SetDefault({});
};
#endif  // SAMPLE_RUNTIME_H_
