/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Objects to describe inference progress
 *************************************************************************/
#ifndef INFERENCE_H_
#define INFERENCE_H_

#include "mm_runtime.h"
#include "common/buffer.h"
#include "mm_run/stage.h"

class Infer {
 public:
  // 结构体SetUp，用于配置Infer类的参数
  struct SetUp {
    IEngine *engine       = nullptr;                       // 指向引擎的指针
    bool copy             = true;                          // 是否进行数据拷贝操作
    bool host_async       = false;                         // 是否在主机上异步执行
    NotifierType tracer   = NotifierType::host;            // 时间通知类型，默认为主机时间通知
    uint64_t bind_bitmap  = 0x00;                          // 设备绑定的位图
    int device_id         = 0;                             // 设备ID
    int infer_depth       = 2;                             // 推理阶段的深度（处理单元个数）
    int buffer_depth      = 2;                             // 缓冲区深度（处理单元个数）
    InferenceTrace *trace = nullptr;                       // 推理跟踪指针，用于记录推理过程
    ShapeGroups shapes;                                    // 形状组，用于存储输入形状组和输出形状组
    std::vector<std::string> input_path{};                 // 输入路径的向量
    std::string output_path{};                             // 输出路径的字符串
    std::string debug_path{};                              // 调试路径的字符串
  };

  // Infer类的构造函数，接受一个SetUp结构体作为参数
  Infer(const SetUp &set);

  // Infer类的析构函数
  ~Infer();

  // 查询当前推理的状态
  void Query();

  // 同步推理结果
  float Sync();

  // 同步所有推理结果
  void SyncAll();

  // 清除推理的跟踪信息
  void ClearTrace();

  // 获取调试信息的字符串表示
  std::string DebugString() const;

 private:
  // 初始化缓冲区
  void InitBuffers();

  // 初始化推理阶段的流水线
  void InitPipeline();

  // 等待Fifo并执行一次出队操作
  void WaitFifoAndPopOnce();

 private:
  SetUp set_;                          // 配置参数结构体
  bool fill_in_           = false;     // 是否填充输入数据
  bool dynamic_shape_     = false;     // 是否使用动态形状
  bool use_dynamic_infer_ = false;     // 是否使用动态推理
  bool can_query_         = false;     // 是否可以进行查询
  BufferGroups in_bufs_;               // 输入缓冲区组
  BufferGroups out_bufs_;              // 输出缓冲区组
  Fifo cpin_fifo_  = Fifo(0);          // 输入数据拷贝Fifo
  Fifo enq_fifo_   = Fifo(0);          // 推理操作Fifo
  Fifo cpout_fifo_ = Fifo(0);          // 输出数据拷贝Fifo
  CpyIn *cpy_in_;                      // 输入数据拷贝的Stage对象
  Enqueue *enq_;                       // 推理操作的Stage对象
  CpyOut *cpy_out_;                    // 输出数据拷贝的Stage对象

 private:
  // MagicMind相关变量
  IContext *context_ = nullptr;                        // 上下文指针
  std::vector<std::vector<Dims>> all_shapes_;          // 所有形状的向量，用于存储输入和输出形状
  std::vector<std::vector<IRTTensor *>> in_tensors_;   // 输入Tensor的向量
  std::vector<std::vector<IRTTensor *>> out_tensors_;  // 输出Tensor的向量
};

#endif  // INFERENCE_H_

