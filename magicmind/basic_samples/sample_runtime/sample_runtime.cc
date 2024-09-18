#include <memory>
#include <dlfcn.h>
#include "cnrt.h"
#include "common/device.h"
#include "common/data.h"
#include "common/logger.h"
#include "basic_samples/sample_runtime/sample_runtime.h"

namespace {

bool IsQuant(magicmind::DataType type) {
  return type == magicmind::DataType::QINT8 || type == magicmind::DataType::QINT16;
}

// Malloc/Free/Copy memory in local/remote.
void *MallocMLUAddr(size_t size,
                    const std::string &remote_server,
                    magicmind::IRpcSession *rpc_sess) {
  void *mem = nullptr;
  if (remote_server.empty()) {
    CHECK_CNRT(cnrtMalloc(&mem, size));
  } else {
    CHECK_STATUS(rpc_sess->Malloc(&mem, size, magicmind::TensorLocation::kRemoteMLU));
  }
  return mem;
}
void FreeMLUAddr(void *ptr, const std::string &remote_server, magicmind::IRpcSession *rpc_sess) {
  if (remote_server.empty()) {
    CHECK_CNRT(cnrtFree(ptr));
  } else {
    CHECK_STATUS(rpc_sess->Free(ptr, magicmind::TensorLocation::kRemoteMLU));
  }
}
}  // namespace

InferInstance::InferInstance(const SetUp &set) : set_(set) {
  // Driver/CNRT back ground
  // Local infer
  if (set_.remote_server.empty()) {
    CHECK_CNRT(cnrtSetDevice(set_.device_id));
    CHECK_CNRT(cnrtQueueCreate(&queue_));
    if (set_.bind_cluster && CheckBindBitmap(set_.device_id, set_.visible_cluster)) {
      uint64_t bitmap = 0;
      for (auto index : set_.visible_cluster) {
        bitmap = bitmap | GenBindBitmap(set_.device_id, index);
      }
      BindCluster(set_.device_id, bitmap);
    }
  } else {
    CHECK_VALID(set_.rpc_sess);
    CHECK_VALID(set_.rpc_sess->DevCount());
    dev_ = set_.rpc_sess->GetDeviceById(set_.device_id);
    CHECK_VALID(dev_);
    queue_rpc_ = dev_->CreateQueue();
    queue_ = queue_rpc_->get();
  }
  // basic objs
  CHECK_VALID(set_.engine);
  CreateIContextAndSetInferWorkSpace();
  MallocIO();
  EnqueueAndDumpAndProfile();
  CopyOut();
}

void InferInstance::CreateIContextAndSetInferWorkSpace() {
  context_ = set_.engine->CreateIContext();
  CHECK_VALID(context_);
  if (set_.mem_stg == kUserStaticWorkSpace) {
    size_t size = 0;
    // WARNING: if network has dynamic output size (which means its outputs
    // are related to its intermediate tensor value), Query will return
    // error::Code::UNAVAILABLE, and can not set static workspace!
    CHECK_STATUS(set_.engine->QueryContextMaxWorkspaceSize(set_.in_dims, &size));
    if (size > 0) {
      intermedia_workspace_addr_ = MallocMLUAddr(size, set_.remote_server, set_.rpc_sess);
      CHECK_STATUS(context_->SetWorkspace(intermedia_workspace_addr_, size));
    }
  }
}

void InferInstance::Memcpy(void *dst, void *src, size_t size, bool host_to_mlu) {
  if (set_.remote_server.empty()) {
    if (host_to_mlu) {
      CHECK_CNRT(cnrtMemcpy(dst, src, size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    } else {
      CHECK_CNRT(cnrtMemcpy(dst, src, size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
  } else {
    if (host_to_mlu) {
      CHECK_STATUS(
          set_.rpc_sess->MemCopy(dst, src, size, magicmind::RpcMemCopyDir::kClientToRemoteMLU));
    } else {
      CHECK_STATUS(
          set_.rpc_sess->MemCopy(dst, src, size, magicmind::RpcMemCopyDir::kRemoteMLUToClient));
    }
  }
}

void InferInstance::MallocIO() {
  if (set_.remote_server.empty()) {
    CHECK_STATUS(context_->CreateInputTensors(&input_tensors_));
    CHECK_STATUS(context_->CreateOutputTensors(&output_tensors_));
  } else {
    CHECK_STATUS(set_.rpc_sess->CreateInputTensors(context_, &input_tensors_));
    CHECK_STATUS(set_.rpc_sess->CreateOutputTensors(context_, &output_tensors_));
  }
  void *ptr = nullptr;
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    CHECK_STATUS(input_tensors_[i]->SetDimensions(set_.in_dims[i]));
    auto size = input_tensors_[i]->GetSize();
    CHECK_CNRT(cnrtHostMalloc(&ptr, size));
    // fill in input data
    if (set_.use_random) {
      std::vector<float> in =
          GenRand<float>(input_tensors_[i]->GetDimensions().GetElementCount(), -1.0, 1.0, 0);
      if (IsQuant(input_tensors_[i]->GetDataType())) {
        auto quant_param = RangeToUniformQuantParamWithQuantAlgV2(
            {-1, 1}, DataTypeSize(input_tensors_[i]->GetDataType()) * 8, "symmetric",
            magicmind::RoundingMode::ROUND_HALF_TO_EVEN);
        CHECK_STATUS(magicmind::QuantizationCast(
            ptr, input_tensors_[i]->GetDataType(), in.data(), magicmind::DataType::FLOAT32,
            in.size(), quant_param, magicmind::RoundingMode::ROUND_HALF_TO_EVEN));
      } else {
        CHECK_STATUS(magicmind::NormalCast(ptr, input_tensors_[i]->GetDataType(), in.data(),
                                           magicmind::DataType::FLOAT32, in.size(), false));
      }
    } else {
      CHECK_VALID(ReadDataFromFile(set_.data_path[i], ptr, size));
    }
    input_host_ptrs_.push_back(ptr);
    // Some network has param as input, Host address will speed them up.
    if ((input_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) ||
        (input_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kRemoteMLU)) {
      CHECK_STATUS(
          input_tensors_[i]->SetData(MallocMLUAddr(size, set_.remote_server, set_.rpc_sess)));
      Memcpy(input_tensors_[i]->GetMutableData(), ptr, size, true);
    } else {
      CHECK_STATUS(input_tensors_[i]->SetData(ptr));
    }
  }
  if (!set_.use_capacity) {
    // WARNING: if network has dynamic output size (which means its outputs
    // are related to its intermediate tensor value), infer will return
    // error::Code::UNAVAILABLE.
    auto status = context_->InferOutputShape(input_tensors_, output_tensors_);
    if (status.code() == magicmind::error::Code::UNAVAILABLE) {
      dynamic_output_ = true;
    } else {
      CHECK_STATUS(status);
    }
  } else {
    CHECK_EQ(output_tensors_.size(), set_.output_capacities.size());
  }
  if (!dynamic_output_) {
    for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
      auto size = output_tensors_[i]->GetSize();
      if (set_.use_capacity) {
        CHECK_STATUS(output_tensors_[i]->SetCapacity(set_.output_capacities[i]));
        size = set_.output_capacities[i];
      }
      CHECK_VALID(
          (output_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) ||
          (output_tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kRemoteMLU));
      CHECK_STATUS(
          output_tensors_[i]->SetData(MallocMLUAddr(size, set_.remote_server, set_.rpc_sess)));
      CHECK_CNRT(cnrtHostMalloc(&ptr, size));
      output_host_ptrs_.push_back(ptr);
    }
  }
}

void InferInstance::EnqueueAndDumpAndProfile() {
  if (set_.do_dump) {
    magicmind::IContext::ContextDumpInfo dump_info;
    // dump all intermedia values
    dump_info.SetDumpMode(magicmind::IContext::ContextDumpInfo::DumpMode::kAllTensors);
    dump_info.SetPath("./sample_dump/");
    dump_info.SetFileFormat(magicmind::IContext::ContextDumpInfo::FileFormat::kText); /*pbtxt*/
    CHECK_STATUS(context_->SetContextDumpInfo(dump_info));
  }
  if (set_.do_profile) {
    magicmind::ProfilerOptions opt;
    opt.SetHostTracerLevel(magicmind::HostTracerLevel::kCritical);
    opt.SetDeviceTracerLevel(magicmind::DeviceTracerLevel::kOn);
    if (set_.remote_server.empty()) {
      profiler_ = magicmind::CreateIProfiler(opt, "./sample_profiler");
    } else {
      profiler_ = set_.rpc_sess->CreateIProfiler(opt, "./sample_profiler");
    }
    CHECK_VALID(profiler_);
    CHECK_VALID(profiler_->Start());
  }
  if (!dynamic_output_) {
    CHECK_STATUS(context_->Enqueue(input_tensors_, output_tensors_, queue_));
  } else {
    CHECK_STATUS(context_->Enqueue(input_tensors_, &output_tensors_, queue_));
    for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
      auto size = output_tensors_[i]->GetSize();
      void *ptr = nullptr;
      CHECK_CNRT(cnrtHostMalloc(&ptr, size));
      output_host_ptrs_.push_back(ptr);
    }
  }
  if (set_.remote_server.empty()) {
    CHECK_CNRT(cnrtQueueSync(queue_));
  } else {
    CHECK_STATUS(queue_rpc_->Sync());
  }
  if (set_.do_profile) {
    CHECK_VALID(profiler_->Stop());
  }
}

void InferInstance::CopyOut() {
  for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
    auto size = output_tensors_[i]->GetSize();
    // fill in input data
    Memcpy(output_host_ptrs_[i], output_tensors_[i]->GetMutableData(), size, false);
  }
}

InferInstance::~InferInstance() {
  // destroy must do strictly as follow
  // destroy tensor/address first
  for (auto ptr : input_host_ptrs_) {
    cnrtFreeHost(ptr);
  }
  for (auto ptr : output_host_ptrs_) {
    cnrtFreeHost(ptr);
  }
  for (auto t : input_tensors_) {
    t->Destroy();
  }
  for (auto t : output_tensors_) {
    t->Destroy();
  }
  // destroy context/profiler
  if (profiler_) {
    profiler_->Destroy();
  }
  context_->Destroy();
  // destroy workspace
  if (set_.mem_stg == kUserStaticWorkSpace) {
    FreeMLUAddr(intermedia_workspace_addr_, set_.remote_server, set_.rpc_sess);
  }
  if (set_.remote_server.empty()) {
    CHECK_CNRT(cnrtQueueDestroy(queue_));
  } else {
    CHECK_STATUS(queue_rpc_->Destroy());
    CHECK_STATUS(dev_->Destroy());
  }
}

void RunInSinglethread(const InferInstance::SetUp &setup) {
  InferInstance infer_obj(setup);
}

RuntimeSample::RuntimeSample(const std::string &model_name,
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
                             const std::vector<std::string> &plugin_libs) {
  model_name_ = model_name;
  in_dims_ = ToDims(input_shapes);
  setup_.in_dims = in_dims_;
  if (input_data.size() == 0) {
    setup_.use_random = true;
  } else {
    setup_.data_path = input_data;
  }
  if (output_capacities.size() > 0) {
    setup_.output_capacities = output_capacities;
  }
  remote_server_ = rpc;
  setup_.remote_server = remote_server_;
  if (remote_server_.empty() && output_capacities.size() > 0) {
    setup_.use_capacity = true;
  }
  mem_stg_ = MemoryStrategy(memory_strategy);
  setup_.do_profile = do_profile;
  setup_.do_dump = do_dump;
  dev_id_ = dev_id;
  setup_.device_id = dev_id_;
  plugins_ = plugin_libs;
  threads_ = threads;
  visible_cluster_ = visible_cluster;
  if (visible_cluster.size() > 0) {
    setup_.bind_cluster = true;
  }
  setup_.mem_stg = mem_stg_;
}

void RuntimeSample::PreparePlugin() {
  for (auto e_ : plugins_) {
    auto kernel_lib = dlopen(e_.c_str(), RTLD_LAZY);
    if (!kernel_lib) {
      SLOG(ERROR) << "Call dlopen() failed : " << dlerror();
      abort();
    }
    plugin_libs_.push_back(kernel_lib);
  }
}

void RuntimeSample::CreateIModelAndQueryInfo() {
  // Local infer
  if (remote_server_.empty()) {
    CHECK_CNRT(cnrtSetDevice(dev_id_));
  } else {
    // Rpc infer for remote-side device
    sess_ = magicmind::RpcConnect(remote_server_);
    setup_.rpc_sess = sess_;
  }
  if (remote_server_.empty()) {
    // nl/plugin/extra kernel will load during createimodel
    KernelMemQuery q("CreateIModel");
    model_ = magicmind::CreateIModel();
  } else {
    model_ = sess_->CreateIModel();
  }
  CHECK_VALID(model_);
  CHECK_STATUS(model_->DeserializeFromFile(model_name_.c_str()));
  ///////////////////Print info below//////////////////////
  size_t size = 0;
  CHECK_STATUS(model_->GetSerializedModelSize(&size));
  SLOG(INFO) << "Create IModel done.";
  SLOG(INFO) << "Name: " << model_name_ << " Size: " << size;
  SLOG(INFO) << "Input num: " << model_->GetInputNum();
  SLOG(INFO) << "Input info:[ ";
  auto names = model_->GetInputNames();
  CHECK_EQ(names.size(), in_dims_.size());
  auto dims = model_->GetInputDimensions();
  auto types = model_->GetInputDataTypes();
  for (uint32_t i = 0; i < names.size(); ++i) {
    SLOG(INFO) << names[i] << ": " << dims[i] << ", " << magicmind::TypeEnumToString(types[i]);
  }
  SLOG(INFO) << "]";
  SLOG(INFO) << "Output info:[ ";
  names = model_->GetOutputNames();
  dims = model_->GetOutputDimensions();
  types = model_->GetOutputDataTypes();
  for (uint32_t i = 0; i < names.size(); ++i) {
    SLOG(INFO) << names[i] << ": " << dims[i] << ", " << magicmind::TypeEnumToString(types[i]);
  }
  SLOG(INFO) << "]";
  ////////////////////////////////////////////////////////
}

void RuntimeSample::CreateIEngineAndSetConstWorkSpace() {
  if (mem_stg_ == MemoryStrategy::kUserStaticWorkSpace) {
    {
      KernelMemQuery e("CreateEngine");
      engine_ = model_->CreateIEngine();
      CHECK_VALID(engine_);
    }
    size_t size = 0;
    CHECK_STATUS(engine_->QueryConstDataSize(&size));
    if (size > 0) {
      const_workspace_addr_ = MallocMLUAddr(size, remote_server_, sess_);
      CHECK_STATUS(engine_->SetConstData(const_workspace_addr_, size));
    }
  } else if (mem_stg_ == MemoryStrategy::kUserDynamicWorkSpace) {
    magicmind::IModel::EngineConfig config;
    static_cast<void>(config.SetDeviceType("MLU"));
    allocator_ = new PointerAdapterAllocator();
    config.SetAllocator(allocator_);
    {
      // Deep fusion kernel will load during createiengine.
      KernelMemQuery e("CreateEngine");
      engine_ = model_->CreateIEngine(config);
      CHECK_VALID(engine_);
    }
  } else {
    {
      KernelMemQuery e("CreateEngine");
      engine_ = model_->CreateIEngine();
      CHECK_VALID(engine_);
    }
  }
  setup_.engine = engine_;
}

void RuntimeSample::DoInfer() {
  PreparePlugin();
  CreateIModelAndQueryInfo();
  CreateIEngineAndSetConstWorkSpace();
  for (int i = 0; i < threads_; i++) {
    setup_.thread_id = i;
    if (setup_.bind_cluster) {
      setup_.visible_cluster = visible_cluster_[i];
    }
    thds_.emplace_back(RunInSinglethread, setup_);
  }
  Destroy();
}

void RuntimeSample::Destroy() {
  for (int i = 0; i < threads_; i++) {
    thds_[i].join();
  }
  // destory engine
  CHECK_STATUS(engine_->Destroy());
  // destroy workspace
  if (mem_stg_ == MemoryStrategy::kUserStaticWorkSpace) {
    FreeMLUAddr(const_workspace_addr_, remote_server_, sess_);
  } else if (mem_stg_ == MemoryStrategy::kUserDynamicWorkSpace) {
    delete allocator_;
  }
  // destroy model
  model_->Destroy();
  // destroy other
  for (auto plugin : plugin_libs_) {
    dlclose(plugin);
  }
  if (!remote_server_.empty()) {
    CHECK_STATUS(sess_->Destroy());
  }
}

int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  InferArg arg_reader;
  arg_reader.ReadIn(args);
  auto data_path = Value(arg_reader.data_path());
  auto model_path = Value(arg_reader.model_path());
  auto input_dims = Value(arg_reader.input_dims());
  auto dev_ids = Value(arg_reader.dev_ids());
  auto plugin_libs = Value(arg_reader.plugin_libs());
  auto mem_stg_param = arg_reader.mem_stg();
  int stg = 0;
  if (HasValue(mem_stg_param)) {
    if (Value(mem_stg_param) == "dynamic") {
      stg = 2;
    } else {
      stg = 1;
    }
  }
  auto profile = Value(arg_reader.profile());
  auto dump = Value(arg_reader.dump());
  auto threads = Value(arg_reader.threads());
  if (threads < 1) {
    SLOG(ERROR) << "The number of threads needs to be greater than 0.";
    abort();
  }
  auto visible_cluster = Value(arg_reader.visible_cluster());
#ifdef __aarch64__
  std::string rpc_server = "";
#else
  auto rpc_server = Value(arg_reader.rpc_server());
  if (!rpc_server.empty() && plugin_libs.size() > 0) {
    SLOG(ERROR) << "MagicMind does not support remote with plugin yet.";
    abort();
  }
  if (visible_cluster.size() > 0 && (!rpc_server.empty())) {
    SLOG(ERROR) << "MagicMind does not support remote with visible cluster.";
    abort();
  }
#endif
  if (visible_cluster.size() > 0 && int(visible_cluster.size()) != threads) {
    SLOG(ERROR) << "The number of threads and clusters do not match.";
    abort();
  }
  auto output_caps = Value(arg_reader.output_caps());
  for (auto id : dev_ids) {
    RuntimeSample sample(model_path, input_dims, data_path, output_caps, rpc_server, stg, threads,
                         profile, dump, visible_cluster, id, plugin_libs);
    sample.DoInfer();
  }
  return 0;
}
