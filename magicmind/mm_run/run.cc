/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include <iomanip>
#include <future>
#include <dlfcn.h>
#include <thread>
#include "common/logger.h"
#include "common/json_util.h"
#include "common/macros.h"
#include "common/device.h"
#include "common/data.h"
#include "common/type.h"
#include "common/timer.h"
#include "mm_run/run.h"


// 匿名命名空间，定义一些内部函数用于线程调用
namespace {
// 单线程运行推理的函数
void RunInSinglethread(const Infer::SetUp &setup,
                       int iterations,
                       float duration,
                       float warmup,
                       const std::string &name,
                       AtomicEvent *start) {
  // 创建Infer对象，用于推理
  Infer infer_obj(setup);
  // 等待启动信号，这里使用了AtomicEvent类来同步线程，保证所有线程在start->Wait(false)之前阻塞，等待start信号后同时开始推理。
  start->Wait(false);
  // 推理预热阶段
  {
    // 创建TimeCollapse对象，用于测量时间
    // std::string("warmup_" + name)是预热阶段的名称，用于在日志中标识预热阶段的耗时
    TimeCollapse time_warm_up(std::string("warmup_" + name));
    float total_time = 0;
    for (; total_time < warmup;) {
      // 进行推理查询
      infer_obj.Query();
      // 同步推理操作，返回当前推理的总时间
      total_time = infer_obj.Sync();
    }
    // 等待所有推理操作完成
    infer_obj.SyncAll();
  }
  // 清除推理的跟踪信息
  infer_obj.ClearTrace();
  // 进行推理
  float total_time = 0;
  {
    // 创建TimeCollapse对象，用于测量时间
    // std::string("infer_" + name)是推理阶段的名称，用于在日志中标识推理阶段的耗时
    TimeCollapse time_infer(std::string("infer_" + name));
    int iter;
    // 设置推理的开始时间，记录在setup.trace->host_start_中
    setup.trace->host_start_ = EnvTime::NowMicros(CLOCK_MONOTONIC);
    for (iter = iterations; iter > 0 || total_time < duration;) {
      // 进行推理查询
      infer_obj.Query();
      // 同步推理操作，返回当前推理的总时间
      auto time = infer_obj.Sync();
      if (time != total_time) {
        --iter;
        total_time = time;
      }
    }
    // 等待所有推理操作完成
    infer_obj.SyncAll();
    // 设置推理的结束时间，记录在setup.trace->host_end_中
    setup.trace->host_end_ = EnvTime::NowMicros(CLOCK_MONOTONIC);
  }
  // 输出推理结果和性能信息到日志
  SLOG(INFO) << "Run " << name << " finished with total duration: " << total_time
             << infer_obj.DebugString();
}


void TraceDevInfo(std::future<bool> start,
                  std::future<bool> stop,
                  const std::vector<int> &dev_ids,
                  std::vector<DeviceUtilInfoTrace> *dev_infos,
                  std::vector<PMUUtilInfoTrace> *pmu_infos,
                  HostUtilInfoTrace *host_infos) {
  PMUCounter *pmu_counter_ = nullptr;
  PMUCounter::PMUData d;
  HostUtilData h = GetHostUtil();
  if (pmu_infos) {
    pmu_counter_ = new PMUCounter();
  }
  while (start.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
    // just wait;
  }
  do {
    auto h_now = GetHostUtil();
    host_infos->emplace_back(h_now - h);
    h = h_now;
    for (size_t i = 0; i < dev_ids.size(); ++i) {
      (*dev_infos)[i].push_back(DeviceUtilInfo(dev_ids[i]));
      if (pmu_infos) {
        if (d.dev_id == -1) {
          d = pmu_counter_->GetUtil(dev_ids[i]);
        } else {
          auto now_d = pmu_counter_->GetUtil(dev_ids[i]);
          (*pmu_infos)[i].push_back(now_d - d);
          d = now_d;
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } while (stop.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
  if (pmu_counter_) {
    delete pmu_counter_;
  }
}

}  // namespace


// Run类的构造函数，接受一个RunParam对象指针作为参数，用于初始化Run类的成员变量
Run::Run(RunParam *param) : params_(param) {
  // 如果需要加载插件，使用dlopen加载插件，并保存句柄到dlhandler_vec_
  if (HasValue(params_->plugin())) {
    auto paths = Value(params_->plugin());
    for (auto path : paths) {
      // 检查路径是否为本地路径，如果不是，则添加本地路径前缀
      path = AddLocalPathIfName(path);
      // 使用dlopen加载插件
      void *handle = dlopen(path.c_str(), RTLD_LAZY);
      // 如果加载失败，打印错误信息并终止程序
      if (!handle) {
        SLOG(ERROR) << "Call dlopen() failed : " << dlerror();
        abort();
      }
      // 将加载的插件句柄保存到dlhandler_vec_中
      dlhandler_vec_.push_back(handle);
    }
  }
  // 检查是否需要禁用数据拷贝，如果禁用了数据拷贝，检查缓冲区深度和推理深度是否大于1
  if (!Value(params_->disable_data_copy())) {
    if ((Value(params_->buffer_depth()) > 1) || (Value(params_->infer_depth()) > 1)) {
      SLOG(WARNING) << "Buffer/Infer depth beyond 1 actually shows weak performance impact when "
                       "disable data copy is on, but doubles the memory footprint.";
      SLOG(WARNING) << "If user is doing memory usage observation, we recommand to set "
                       "infer/buffer depth to 1.";
    }
  }
  // 获取设备ID列表并检查合法性
  dev_ids_ = Value(params_->devices());
  // 检查设备ID是否合法，设备ID不能大于系统中可用的设备数
  uint32_t num_device = 0;
  CHECK_CNRT(cnrtGetDeviceCount(&num_device));
  for (size_t i = 0; i < dev_ids_.size(); i++) {
    uint32_t dev_id = dev_ids_[i];
    if (num_device - 1 < dev_id) {
      SLOG(ERROR) << "Invalid dev id " << dev_id << ", dev_num is:" << num_device;
    }
  }
  // 设置当前活动设备为dev_ids_中的第一个设备
  CHECK_CNRT(cnrtSetDevice(dev_ids_[0]));
  // 反序列化模型
  {
    TimeCollapse time_create_model("DeserializeModel");
#ifdef NDEBUG
    KernelMemQuery mem_occ("CreateModel");
#endif
    // 创建IModel对象并从文件中反序列化模型
    model_ = CreateIModel();
    CHECK_STATUS(model_->DeserializeFromFile(Value(params_->magicmind_model()).c_str()));
  }
  // 初始化模型输入输出形状信息
  InitShapes();
  // 获取模型信息，并输出模型的输入输出信息
  ModelInfo();
  // 创建并初始化模型引擎
  InitEngines();
  // 创建并初始化线程池，用于多线程推理
  InitThreads();
  // 释放模型以节省资源，因为模型已经反序列化并创建了引擎
  if (model_) {
    model_->Destroy();
    model_ = nullptr;
  }
}


// 输出模型的输入输出信息
void Run::ModelInfo() {
  std::stringstream info;
  size_t size = 0;
  // 获取模型的序列化大小
  CHECK_STATUS(model_->GetSerializedModelSize(&size));
  // 输出模型的基本信息
  info << "=================== Model Information" << std::endl;
  info << std::setw(10) << std::left << "Size: " << size << std::endl;
  info << std::setw(10) << std::left << "Input num: " << model_->GetInputNum() << std::endl;
  info << std::setw(10) << std::left << "Output num: " << model_->GetOutputNum() << std::endl;
  // 输出模型的输入信息
  info << "Input info [" << std::endl;
  auto names = model_->GetInputNames();
  // 检查模型的输入名称数与形状数是否一致
  CHECK_EQ(names.size(), shapes_[0].size());
  auto dims = model_->GetInputDimensions();
  auto types = model_->GetInputDataTypes();
  for (uint32_t i = 0; i < names.size(); ++i) {
    // 输出每个输入的名称、形状和数据类型
    info << std::setw(20) << std::left << names[i] << ": " << dims[i] << ", " << TypeEnumToString(types[i]) << std::endl;
    // 如果某个输入的形状中包含未知元素个数（小于0），则标记该输入为可变输入
    if (dims[i].GetElementCount() < 0) {
      mutable_in_ = true;
    }
  }
  info << "]" << std::endl;
  // 输出模型的输出信息
  info << "Output info [" << std::endl;
  names = model_->GetOutputNames();
  dims = model_->GetOutputDimensions();
  types = model_->GetOutputDataTypes();
  for (uint32_t i = 0; i < names.size(); ++i) {
    // 输出每个输出的名称、形状和数据类型
    info << std::setw(20) << std::left << names[i] << ": " << dims[i] << ", " << TypeEnumToString(types[i]) << std::endl;
    // 如果某个输出的形状中包含未知元素个数（小于0），则标记该输出为可变输出
    if (dims[i].GetElementCount() < 0) {
      mutable_out_ = true;
    }
  }
  info << "]";
  // 将模型信息输出到日志
  SLOG(INFO) << "\n" << info.str();
}


// 设置输入形状
void Run::InitShapes() {
  // 通过配置文件设置输入形状
  if (HasValue(params_->run_config())) {
    // 如果配置文件中存在run_config参数，则使用配置文件中的形状来设置输入
    SLOG(INFO) << "Run config will override all other input shapes.";
    auto config_path = Value(params_->run_config());
    json11::Json obj;
    CHECK_VALID(ReadJsonFromFile(config_path, &obj));
    shapes_ = ShapeGroups(obj);
  } else {
    // 否则，默认使用模型的输入形状设置输入
    auto model_inputs = model_->GetInputDimensions();
    std::vector<std::vector<int>> input_shapes;
    // 遍历模型的输入，获取每个输入的形状，并将其转换为std::vector<int>形式存储在input_shapes中
    for (auto in : model_inputs) {
      auto s_ = in.GetDims();
      input_shapes.push_back(std::vector<int>({s_.begin(), s_.end()}));
    }
    if (HasValue(params_->input_dims())) {
      // 如果用户在命令行指定了input_dims参数，则使用用户指定的形状来设置输入
      auto v_ = Value(params_->input_dims());
      input_shapes = v_;
    }
    if (HasValue(params_->batch_size())) {
      // 如果用户在命令行指定了batch_size参数，则将用户指定的batch size应用到相应的输入形状中
      auto batchs_ = Value(params_->batch_size());
      CHECK_EQ(batchs_.size(), input_shapes.size());
      for (size_t i = 0; i < input_shapes.size(); ++i) {
        if (input_shapes[i].size() > 0) {
          input_shapes[i][0] = batchs_[i];
        }
      }
    }
    // 使用整理后的输入形状创建ShapeGroups对象，并作为输入形状的设置
    shapes_ = ShapeGroups(std::vector<std::vector<std::vector<int>>>({input_shapes}));
  }
  // 如果输入形状中有名称信息，则重新排序ShapeGroups对象，以与模型的输入名称相匹配
  if (shapes_.has_name()) {
    auto name_vec = model_->GetInputNames();
    shapes_.Reorder(name_vec);
  }
}


// 创建推理引擎
void Run::InitEngines() {
  // 为每个设备创建推理引擎
  dev_infos_.resize(dev_ids_.size());
  pmu_infos_.resize(dev_ids_.size());
  traces_.resize(dev_ids_.size());
  engine_.resize(dev_ids_.size());
  // 配置推理引擎
  static_cast<void>(config_.SetDeviceType("MLU"));
  config_.SetConstDataInit(true);
  config_.SetKernelCapture(Value(params_->kernel_capture()));
  // 创建推理引擎向量
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    // 设置当前设备
    CHECK_CNRT(cnrtSetDevice(dev_ids_[i]));
    DeviceInfo dev_info;
    SLOG(INFO) << dev_info;
    // 如果需要绑定Cluster，必须在创建引擎之前启用该功能，
    // 否则在实际运行中，可能导致内核行为发生变化，而实际运行时的工作空间大小不一致。
    bind_cluster_ = Value(params_->bind_cluster());
    if (bind_cluster_) {
      // 初始化设备环境并绑定Cluster
      InitDeviceEnv();
      BindCluster(dev_ids_[i], 0x01);  // 0是必需的，用于设置Cluster。
    }
    {
      // 创建引擎并记录创建引擎所用的时间
      TimeCollapse time_create_engine("CreateEngine " + std::to_string(i));
#ifdef NDEBUG
      KernelMemQuery mem_occ("CreateEngine " + std::to_string(i));
#endif
      engine_[i] = model_->CreateIEngine(config_);
      CHECK_VALID(engine_[i]);
    }
    // 查询常量数据的大小并输出到日志
    uint64_t const_size = 0;
    CHECK_STATUS(engine_[i]->QueryConstDataSize(&const_size));
    SLOG(INFO) << "Const data size: " << const_size * 1.0 / 1024 / 1024 << "(MB)";
    // 查询工作空间的大小并输出到日志
    uint64_t work_space = 0;
    bool has_workspace = true;
    for (size_t shape_idx = 0; shape_idx < shapes_.size(); ++shape_idx) {
      uint64_t work_space_query = 0;
      auto shape = shapes_[shape_idx];
      std::vector<std::vector<int>> query_shape = shape.GetShapes();
      auto ret = engine_[i]->QueryContextMaxWorkspaceSize(ToDims(query_shape), &work_space_query);
      if (!ret.ok()) {
        CHECK_VALID((ret.code() == error::Code::UNAVAILABLE));
        SLOG(INFO) << "ContextMaxworkspace Size: UNAVAILABLE";
        has_workspace = false;
        break;
      }
      work_space = work_space_query > work_space ? work_space_query : work_space;
    }
    if (has_workspace) {
      SLOG(INFO) << "ContextMaxworkspace Size: " << work_space * 1.0 / 1024 / 1024 << "(MB)";
    }
  }
}


// 初始化线程池和追踪信息
void Run::InitThreads() {
  // 获取线程数量
  thread_num_ = Value(params_->threads());
  // 为每个设备创建线程池，并初始化对应的追踪信息
  for (size_t i = 0; i < engine_.size(); ++i) {
    // 创建线程池并将其添加到pools_向量中
    pools_.push_back(new ThreadPool(thread_num_));
    // 为当前设备的每个线程初始化追踪信息
    traces_[i].resize(thread_num_);
    for (auto &t : traces_[i]) {
      // 为每个线程的追踪信息预留空间
      t.input_indexs_.reserve(Value(params_->iterations()));
      t.time_traces_[0].reserve(Value(params_->iterations()));
      t.time_traces_[1].reserve(Value(params_->iterations()));
      t.time_traces_[2].reserve(Value(params_->iterations()));
    }
  }
}


// 多设备并发推理函数
void Run::RunInMultiDevices() {
  // 初始化推理参数 set
  Infer::SetUp set;
  set.copy = !Value(params_->disable_data_copy()); // 是否进行数据拷贝
  set.host_async = Value(params_->host_async()); // 是否使用异步host推理
  set.tracer = StringToNType(Value(params_->trace_time())); // 设置推理追踪器的类型
  set.shapes = shapes_; // 设置推理输入的形状信息
  CHECK_LE(1, Value(params_->buffer_depth())); // 确保buffer_depth至少为1
  CHECK_LE(1, Value(params_->infer_depth())); // 确保infer_depth至少为1
  set.buffer_depth = Value(params_->buffer_depth()); // 设置buffer的深度
  set.infer_depth = Value(params_->infer_depth()); // 设置infer的深度
  set.input_path = Value(params_->input_files()); // 输入数据路径
  set.output_path = Value(params_->output_path()); // 输出结果路径
  set.debug_path = Value(params_->debug_path()); // 调试信息路径
  // 同步信号
  std::vector<std::future<void>> results(dev_ids_.size() * thread_num_); // 存储每个设备和线程的推理任务的future
  std::promise<bool> start_signal; // 启动信号
  std::promise<bool> stop_signal; // 终止信号
  AtomicEvent start_event; // 启动事件
#ifdef USE_PROFILER
  // Profiler
  if (HasValue(params_->perf_path())) {
    ProfilerOptions options;
    options.SetHostTracerLevel(magicmind::HostTracerLevel::kCritical); // 设置Host端追踪器级别为关键级别
    options.SetDeviceTracerLevel(magicmind::DeviceTracerLevel::kOn); // 设置Device端追踪器级别为开启状态
    profiler_ = CreateIProfiler(options, Value(params_->perf_path()).c_str()); // 创建Profiler
    CHECK_VALID(profiler_);
  }
#endif  // USE_PROFILER
  // 遍历每个设备
  for (size_t i = 0; i < dev_ids_.size(); ++i) {
    set.device_id = dev_ids_[i]; // 设置当前设备ID
    set.engine = engine_[i]; // 设置当前设备的推理引擎
    for (int thread_idx = 0; thread_idx < pools_[i]->GetThreadPoolSize(); ++thread_idx) {
      if (bind_cluster_) {
        set.bind_bitmap = GenBindBitmap(dev_ids_[i], thread_idx); // 根据绑定策略生成bitmap
      }
      set.trace = &(traces_[i][thread_idx]); // 设置当前设备和线程的追踪信息
      results[thread_idx] = pools_[i]->AddTask(RunInSinglethread, set, Value(params_->iterations()),
                                               Value(params_->duration()), Value(params_->warmup()),
                                               std::string("dev_" + std::to_string(dev_ids_[i]) +
                                                           "_thread_" + std::to_string(thread_idx)),
                                               &start_event); // 将推理任务添加到线程池中
    }
  }
  std::vector<PMUUtilInfoTrace> *pmu_tracer_ = nullptr;
  if (Value(params_->trace_pmu())) {
    pmu_tracer_ = &pmu_infos_; // 如果开启了PMU追踪，将PMU追踪信息传递给pmu_tracer_
  }
  // 启动追踪设备信息的线程
  std::thread trace_dev(TraceDevInfo, start_signal.get_future(), stop_signal.get_future(), dev_ids_,
                        &dev_infos_, pmu_tracer_, &host_infos_);
#ifdef USE_PROFILER
  if (profiler_) {
    CHECK_VALID(profiler_->Start()); // 启动Profiler
  }
#endif  // USE_PROFILER
  // 发送启动信号，并启动推理任务
  start_signal.set_value(true);
  start_event.PlaceOn();
  // 等待所有推理任务完成
  for (size_t i = 0; i < results.size(); ++i) {
    results[i].get();
  }
  SLOG(INFO) << "Inference finished."; // 推理完成
  // 发送终止信号，并等待追踪设备信息的线程完成
  stop_signal.set_value(true);
  trace_dev.join();
  SLOG(INFO) << "Dev info trace finished."; // 设备信息追踪完成
#ifdef USE_PROFILER
  if (profiler_) {
    profiler_->Stop(); // 停止Profiler
  }
#endif  // USE_PROFILER
}


Run::~Run() {
  // Step 1: 如果开启了推理追踪功能（trace_path不为空），则创建保存追踪结果的文件夹。
  if (HasValue(params_->trace_path())) {
    CHECK_VALID(CreateFolder(Value(params_->trace_path())));
  }
  // Step 2: 输出信息 "Generating report..."，表示正在生成推理报告。
  SLOG(INFO) << "Generating report...";
  // Step 3: 调用Report类构造函数，传入shapes、dev_ids、traces、dev_infos、pmu_infos、host_infos、trace_time和avg_runs参数，生成推理报告report。
  auto report = Report(shapes_, dev_ids_, traces_, dev_infos_, pmu_infos_, host_infos_,
                       StringToNType(Value(params_->trace_time())), Value(params_->avg_runs()));
  // Step 4: 调用report的Print()方法，打印推理报告的概要信息。
  report.Print();
  // Step 5: 调用report的Analysis()方法，根据推理报告进行性能分析，得出性能数据。
  report.Analysis(Value(params_->host_async()), mutable_in_, mutable_out_,
                  Value(params_->buffer_depth()), Value(params_->infer_depth()));
  // Step 6: 如果开启了推理追踪功能，则将report转换为Json格式，并保存到文件中，文件名包含设备ID和当前时间。
  if (HasValue(params_->trace_path())) {
    auto jsons = report.ToJsons();
    for (size_t idx = 0; idx < jsons.size(); ++idx) {
      std::string file = "dev" + std::to_string(dev_ids_[idx]) + EnvTime::CurrentTime() + ".json";
      CHECK_VALID(WriteJsonToFile(Value(params_->trace_path()) + "/" + file, jsons[idx]));
    }
  }
  // Step 7: 如果开启了Profiler功能，则销毁Profiler对象。
#ifdef USE_PROFILER
  if (profiler_) {
    profiler_->Destroy();
  }
#endif  // USE_PROFILER
  // Step 8: 遍历engine_容器，销毁每个引擎对象。
  for (auto e : engine_) {
    if (e) {
      CHECK_STATUS(e->Destroy());
    }
  }
  // Step 9: 清空engine_容器。
  engine_.clear();
  // Step 10: 遍历pools_容器，释放每个线程池对象。
  for (auto t : pools_) {
    delete t;
  }
  // Step 11: 清空pools_容器。
  pools_.clear();
  // Step 12: 遍历dlhandler_vec_容器，关闭每个动态链接库句柄。
  for (auto handle : dlhandler_vec_) {
    dlclose(handle);
  }
}
