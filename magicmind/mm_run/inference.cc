/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Objects to describe inference progress
 *************************************************************************/
#include "common/logger.h"
#include "common/macros.h"
#include "common/data.h"
#include "common/timer.h"
#include "common/type.h"
#include "mm_run/inference.h"

// Infer类的构造函数，接受一个SetUp结构体作为参数
Infer::Infer(const SetUp &set) : set_(set) {
  // 设置设备ID
  CHECK_CNRT(cnrtSetDevice(set_.device_id));
  // 绑定设备和线程
  if (set_.bind_bitmap > 0) {
    BindCluster(set_.device_id, set_.bind_bitmap);
  }
  // 初始化基本对象
  CHECK_VALID(set_.engine);
  CHECK_VALID(set_.trace);
  {
    // 记录创建上下文的时间
    TimeCollapse time_context("CreateContext");
    // 创建推理上下文
    context_ = set_.engine->CreateIContext();
  }
  CHECK_VALID(context_);
  // 如果设置了调试路径，则设置上下文的输出Dump信息
  if (!set_.debug_path.empty()) {
    IContext::ContextDumpInfo dump_info;
    dump_info.SetDumpMode(IContext::ContextDumpInfo::DumpMode::kAllTensors);  // 输出所有中间值
    dump_info.SetPath(set_.debug_path);
    dump_info.SetFileFormat(IContext::ContextDumpInfo::FileFormat::kText);    // 输出格式为pbtxt
    CHECK_STATUS(context_->SetContextDumpInfo(dump_info));
  }
  // 初始化缓冲区
  InitBuffers();
  // 初始化推理阶段的流水线
  InitPipeline();
}

// 初始化缓冲区
void Infer::InitBuffers() {
  // 至少需要有一个形状组
  CHECK_LE(1, set_.shapes.size());
  // 如果有多个形状组，则使用动态形状
  if (set_.shapes.size() > 1) {
    dynamic_shape_ = true;
  }
  // 将形状组转换为Dims，并存储在all_shapes_中
  for (size_t i = 0; i < set_.shapes.size(); ++i) {
    all_shapes_.push_back(ToDims(set_.shapes[i].GetShapes()));
  }
  out_tensors_.resize(set_.infer_depth);
  // 初始化输入缓冲区组
  for (int i = 0; i < set_.buffer_depth; ++i) {
    std::vector<IRTTensor *> ins;
    // 创建输入缓冲区组
    in_bufs_.push_back(new Buffers("InputBufferGroup" + std::to_string(i), set_.copy));
    // 创建输入Tensor
    CHECK_STATUS(context_->CreateInputTensors(&(ins)));
    // 设置输入Tensor的形状
    SetShapes(ins, all_shapes_[0]);
    in_tensors_.push_back(ins);
    // 初始化输入缓冲区
    in_bufs_[i]->Init(ins);
    // 如果设置了数据拷贝且有输入路径，则填充输入数据
    if (set_.copy && set_.input_path.size()) {
      if (!dynamic_shape_) {
        fill_in_ = true;
        in_bufs_[i]->FillIn(set_.input_path);
      } else {
        // 动态形状下不支持填充真实的输入数据
        SLOG(WARNING) << "Fill in real inputs with dynamic input shapes is currently unsupported.";
      }
    }
  }
  // 初始化输出缓冲区组
  for (int i = 0; i < set_.infer_depth; ++i) {
    // 创建输出缓冲区组
    out_bufs_.push_back(new Buffers("OutputBufferGroup" + std::to_string(i), set_.copy));
    // 使用第一个输入Tensor作为参考，创建输出Tensor
    std::vector<IRTTensor *> ins = in_tensors_[0];  // 实际应该是i，但会不会是因为in_tensors_中保存的shape都相同，所以只用in_tensors_[0]，这样的话就可以将其存入cache中提高速度，而不用每次都去内存中读取i中的数据
    auto status_create = context_->CreateOutputTensors(&out_tensors_[i]);
    if (status_create.ok()) {
      // 推理输出Tensor的形状根据输入Tensor进行推导
      auto status_infer_shape = context_->InferOutputShape(ins, out_tensors_[i]);
      if (status_infer_shape.ok()) {
        if (!dynamic_shape_) {
          // 如果不使用动态形状，则初始化输出缓冲区
          out_bufs_[i]->Init(out_tensors_[i]);
          use_dynamic_infer_ = false;
        } else {
          // 使用动态形状
          use_dynamic_infer_ = true;
        }
      } else if (status_infer_shape.code() == error::Code::UNAVAILABLE) {
        // 如果推导形状不可用，则使用动态形状
        use_dynamic_infer_ = true;
      } else {
        // 推导形状失败，抛出异常
        CHECK_STATUS(status_infer_shape);
      }
    }
    if (status_create.code() == error::Code::UNAVAILABLE) {
      // 如果创建输出Tensor不可用，则使用动态形状
      use_dynamic_infer_ = true;
    } else {
      // 创建输出Tensor失败，抛出异常
      CHECK_STATUS(status_create);
    }
  }
}

// 初始化推理阶段的流水线
void Infer::InitPipeline() {
  // 创建输入和输出Fifo
  cpin_fifo_ = Fifo(set_.buffer_depth);
  cpout_fifo_ = Fifo(set_.infer_depth);
  enq_fifo_ = Fifo(set_.infer_depth);
  // 创建拷贝输入、推理和拷贝输出三个阶段对象
  cpy_in_ = new CpyIn(set_.buffer_depth, set_.host_async, set_.copy, set_.tracer, &cpin_fifo_,
                      &in_bufs_, all_shapes_, &(set_.trace->time_traces_[0]));
  cpy_out_ = new CpyOut(set_.infer_depth, set_.host_async, set_.copy, set_.tracer, &enq_fifo_,
                        &cpout_fifo_, &out_bufs_, &(set_.trace->time_traces_[2]));
  enq_ = new Enqueue(set_.infer_depth, set_.host_async, use_dynamic_infer_, dynamic_shape_,
                     set_.tracer, &cpin_fifo_, &enq_fifo_, &in_bufs_, &out_bufs_, context_, set_.device_id,
                     set_.bind_bitmap, &(set_.trace->time_traces_[1]));
}

// 执行推理查询
void Infer::Query() {
  // 设置可以进行查询
  can_query_ = true;
  // 依次执行拷贝输入、推理和拷贝输出三个阶段，并判断是否可以进行查询
  can_query_ = cpy_in_->DoStage() && can_query_;
  can_query_ = enq_->DoStage() && can_query_;
  can_query_ = cpy_out_->DoStage() && can_query_;
}

// 等待输出Fifo并执行一次出队操作
void Infer::WaitFifoAndPopOnce() {
  // 从输出Fifo中取出一个元素
  auto u = cpout_fifo_.pop();
  // 如果设置了Host异步拷贝
  if (set_.host_async) {
    if (set_.copy) {
      // 如果设置了拷贝，则等待Host拷贝完成
      u.host_->Wait();
    } else {
      // 如果没有设置拷贝，则先等待Host拷贝，再等待设备拷贝完成
      u.host_->Wait();
      u.dev_->Wait();
    }
  } else {
    // 如果没有设置Host异步拷贝，则等待设备拷贝完成
    u.dev_->Wait();
  }
  // 记录时间和索引，重置Stage的状态
  set_.trace->input_indexs_.push_back(u.shape_idx_);
  cpy_in_->CollectTime(u.pipe_idxes_[0]);
  enq_->CollectTime(u.pipe_idxes_[1]);
  cpy_out_->CollectTime(u.pipe_idxes_[2]);
  cpy_in_->ResetActive(u.pipe_idxes_[0]);
  enq_->ResetActive(u.pipe_idxes_[1]);
  cpy_out_->ResetActive(u.pipe_idxes_[2]);
}

// 同步一次推理操作
float Infer::Sync() {
  // 如果可以进行查询且输出Fifo不为空，则等待输出Fifo并执行一次出队操作
  if (can_query_ && cpout_fifo_.size()) {
    WaitFifoAndPopOnce();
  }
  // 返回推理持续时间
  return enq_->LastDuration();
}

// 同步所有推理操作
void Infer::SyncAll() {
  // 循环执行拷贝输入、推理和拷贝输出三个阶段直至输入Fifo为空
  while (cpin_fifo_.size()) {
    can_query_ = true;
    can_query_ = enq_->DoStage() && can_query_;
    can_query_ = cpy_out_->DoStage() && can_query_;
    if (can_query_ && cpout_fifo_.size()) {
      // 如果可以进行查询且输出Fifo不为空，则等待输出Fifo并执行一次出队操作
      WaitFifoAndPopOnce();
    }
  }
  // 循环执行拷贝输出阶段直至输出Fifo为空
  while (cpout_fifo_.size()) {
    // 等待输出Fifo并执行一次出队操作
    WaitFifoAndPopOnce();
  }
}

// 清除推理的跟踪信息
void Infer::ClearTrace() {
  // 清除跟踪信息
  set_.trace->input_indexs_.clear();
  set_.trace->time_traces_[0].clear();
  set_.trace->time_traces_[1].clear();
  set_.trace->time_traces_[2].clear();
  // 重置enqueue的持续时间
  enq_->SetDurationStart();
}

// Infer类的析构函数
Infer::~Infer() {
  // 销毁输入和输出Tensor
  for (auto vec : in_tensors_) {
    for (auto t : vec) {
      t->Destroy();
    }
  }
  for (auto vec : out_tensors_) {
    for (auto t : vec) {
      t->Destroy();
    }
  }
  // 如果填充了输入数据，则输出结果
  if (fill_in_) {
    SLOG(INFO) << "Dump files...";
    out_bufs_[0]->FillOut(set_.output_path);
  }
  // 释放缓冲区对象
  for (auto p : in_bufs_) {
    delete p;
  }
  for (auto p : out_bufs_) {
    delete p;
  }
  // 释放Stage对象和推理上下文
  delete cpy_in_;
  delete enq_;
  delete cpy_out_;
  context_->Destroy();
}

// 获取调试信息的字符串表示
std::string Infer::DebugString() const {
  std::stringstream ret;
  ret << "\n========= Input Buffer Info =========";
  for (size_t idx = 0; idx < in_bufs_.size(); ++idx) {
    ret << in_bufs_[idx]->DebugString();
  }
  ret << "\n========= Output Buffer Info =========";
  for (size_t idx = 0; idx < out_bufs_.size(); ++idx) {
    ret << out_bufs_[idx]->DebugString();
  }
  return ret.str();
}
