/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#ifndef RUN_H_
#define RUN_H_
#ifdef USE_PROFILER
#include "mm_profiler.h" // 包含使用性能分析的头文件
#endif  // USE_PROFILER
#include "common/threadpool.h" // 包含通用线程池的头文件
#include "mm_run/run_param.h" // 包含运行时参数的头文件
#include "mm_run/inference.h" // 包含推理（Inference）的头文件

/*!
 * @class MMRun
 *
 * @brief MMRun类用于初始化IModel并持有IEngine的资源。
 * 具体而言，该类会对带有数据和图的模型进行反序列化，然后在特定设备上创建IEngine，
 * 然后在每个设备上启动多个线程进行推理（Inference）。
 *
 * @par 要求
 *  - mm_runtime.h (寒武纪的Runtime库头文件)
 *  - threadpool.h (通用线程池的头文件)
 */
class Run {
 public:
  // 构造函数，接受RunParam类型的参数指针
  Run(RunParam *params);

  // 在多个设备上执行推理（Inference）的方法
  void RunInMultiDevices();

  // 析构函数
  ~Run();

 private:
  // 初始化模型的形状信息
  void InitShapes();

  // 初始化模型引擎（IEngine）
  void InitEngines();

  // 初始化线程池
  void InitThreads();

  // 获取模型信息的函数
  void ModelInfo();

 private:
  RunParam *params_ = nullptr; // 运行时参数指针
  IModel *model_ = nullptr; // IModel指针
#ifdef USE_PROFILER
  IProfiler *profiler_ = nullptr; // 使用性能分析的Profiler指针
#endif  // USE_PROFILER
  bool mutable_in_ = false; // 可变输入标志
  bool mutable_out_ = false; // 可变输出标志
  IModel::EngineConfig config_; // IModel引擎配置
  std::vector<int> dev_ids_; // 设备ID列表
  std::vector<IEngine *> engine_; // IEngine指针的向量
  std::vector<ThreadPool *> pools_; // 线程池指针的向量
  ShapeGroups shapes_; // 形状组信息
  int thread_num_ = 0; // 线程数
  bool bind_cluster_ = false; // 绑定集群标志
  std::vector<void *> dlhandler_vec_; // 动态链接库句柄向量
  std::vector<DeviceUtilInfoTrace> dev_infos_; // 设备信息的跟踪向量
  std::vector<PMUUtilInfoTrace> pmu_infos_; // 性能计数器信息的跟踪向量
  HostUtilInfoTrace host_infos_; // 主机信息的跟踪结构体
  std::vector<InferenceTraceContainer> traces_; // 推理（Inference）跟踪容器向量
};

#endif  // RUN_H_

