/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "common/logger.h"
#include "common/macros.h"
#include "common/data.h"
#include "common/timer.h"
#include "mm_run/stage.h"

// Stage类的构造函数
Stage::Stage(int depth,
             bool on_host,
             bool skip,
             NotifierType t,
             Fifo *in_fifo,
             Fifo *out_fifo,
             BufferGroups *in_group,
             BufferGroups *out_group,
             TimeInfoContainer *tracer)
    : in_fifo_(in_fifo),                       // 输入Fifo指针
      out_fifo_(out_fifo),                     // 输出Fifo指针
      on_host_(on_host),                       // 是否在主机上执行
      skip_(skip),                             // 是否跳过阶段
      t_(t),                                   // 时间通知类型
      depth_(depth),                           // 处理单元个数
      in_group_(in_group),                     // 输入缓冲区组
      out_group_(out_group),                   // 输出缓冲区组
      total_time_container_(tracer) {          // 时间信息容器指针
  active_ = std::vector<bool>(depth, false);   // 用于记录每个处理单元的活动状态
  queue_ = new Queue();                       // 用于设备间数据传输的队列
  for (int idx = 0; idx < depth; ++idx) {
    // 为每个处理单元创建两个Notifier实例，并保存在notifiers_中
    notifiers_.push_back(
        std::array<Notifier *, 2>({new Notifier(!RecordHost(t_)), new Notifier(!RecordHost(t_))}));
  }
  if (on_host_) {
    // 如果在主机上执行，创建一个线程池和事件容器
    pool_ = new ThreadPool(1);               // 线程池，用于主机上的并行处理
    for (int idx = 0; idx < depth; ++idx) {
      // 为每个处理单元创建一个AtomicEvent实例，并保存在events_中
      events_.push_back(new AtomicEvent());  // 原子事件，用于跟踪主机上的拷贝操作
    }
  }
  current_tps_.resize(depth);                 // 记录每个处理单元的时间戳
}

// Stage类的析构函数
Stage::~Stage() {
  if (queue_) {
    delete queue_;
  }
  if (pool_) {
    delete pool_;
  }
  // 释放notifiers_中的资源
  for (auto e_ : notifiers_) {
    delete e_[0];
    delete e_[1];
  }
  // 释放events_中的资源
  for (auto e_ : events_) {
    delete e_;
  }
}

// 执行当前阶段的处理操作
bool Stage::DoStage() {
  if (active_[current_index_]) {
    return true;      // 如果当前处理单元已经活动，直接返回
  }
  if (in_fifo_ && in_fifo_->size() == 0) {
    return true;      // 如果输入Fifo为空，直接返回
  }
  active_[current_index_] = true;   // 将当前处理单元状态设置为活动
  FifoUnit data;
  if (in_fifo_) {
    data = in_fifo_->pop();         // 从输入Fifo中取出数据
  }
  // 调用派生类的DoWork()方法进行具体处理
  DoWork(data);
  // 切换到下一个处理单元
  MoveNext();
  return skip_;                     // 返回是否跳过阶段
}

// 切换到下一个处理单元
void Stage::MoveNext() {
  current_index_ = (current_index_ + 1) % depth_;  // 循环切换到下一个处理单元
}

// 重置指定索引的处理单元的状态
void Stage::ResetActive(int index) {
  active_[index] = false;       // 重置指定索引的处理单元状态为非活动
}

// CpyIn类的构造函数，继承了Stage类的构造函数
CpyIn::CpyIn(int buffer_depth,
             bool on_host,
             bool do_cpy,
             NotifierType t,
             Fifo *cpy_in_fifo,
             BufferGroups *in_group,
             const std::vector<std::vector<Dims>> &all_shapes,
             TimeInfoContainer *tracer)
    : Stage(buffer_depth, on_host, !do_cpy, t, nullptr, cpy_in_fifo, in_group, nullptr, tracer),
      all_shapes_(all_shapes) {
  if (on_host_) {
    // 在主机上创建一个函数对象(host_cpy_)，用于主机到设备的拷贝操作
    host_cpy_ = [this](Buffers *buffer, int idx) {
      current_tps_[idx].first = EnvTime::NowMicros(CLOCK_MONOTONIC);  // 记录拷贝开始时间
      buffer->H2D();                 // 主机到设备的数据拷贝操作
      current_tps_[idx].second = EnvTime::NowMicros(CLOCK_MONOTONIC); // 记录拷贝结束时间
      // place host event
      events_[idx]->PlaceOn();      // 将事件放置在队列中，用于跟踪处理时间
    };
  }
};

// 收集时间信息，并保存到total_time_container_中
void CpyIn::CollectTime(int index) {
  // 收集时间信息
  if (!skip_) {
    float interface_duration_ =
        float(current_tps_[index].second - current_tps_[index].first) / 1000; // 拷贝持续时间
    if (on_host_) {
      // 如果在主机上执行，将时间信息添加到时间信息容器中
      total_time_container_->emplace_back(interface_duration_, interface_duration_,
                                          interface_duration_);
    } else {
      // 如果在设备上执行，将时间信息添加到时间信息容器中
      total_time_container_->emplace_back(
          RecordHost(t_) ? notifiers_[index][1]->HostTimeFrom(*notifiers_[index][0]) : 0,
          RecordDev(t_) ? notifiers_[index][1]->DevTimeFrom(*notifiers_[index][0]) : 0,
          interface_duration_);
    }
  }
}

// 实现数据拷贝操作
void CpyIn::DoWork(const FifoUnit &unit) {
  FifoUnit out;
  out.pipe_idxes_[0] = current_index_;
  out.shape_idx_ = shape_idx_;
  // prepare shape
  auto buffers = (*in_group_)[current_index_]; // 获取当前处理单元的缓冲区组
  MoveShapeNext();                            // 切换到下一个shape
  if (out.shape_idx_ != shape_idx_) {
    // 如果切换到了不同的shape，重新设置shape并重新初始化缓冲区组
    SetShapes(buffers->OriTensors(), all_shapes_[out.shape_idx_]);
    buffers->ReInit();
  }
  // 执行具体的拷贝操作
  if (!skip_) {
    if (on_host_) {
      // 如果在主机上执行，使用线程池进行主机到设备的拷贝操作
      pool_->AddTask(host_cpy_, buffers, current_index_);
      // cpy on host, so enqueue should wait for host thread
      out.host_ = events_[current_index_];    // 将事件添加到输出Fifo中，用于等待主机线程的结束
    } else {
      // 如果在设备上执行，执行设备到设备的数据拷贝，并使用时间通知器记录时间
      if (t_ != NotifierType::none) {
        // 如果指定了时间通知类型，将notifier置于队列中
        notifiers_[current_index_][0]->PlaceOn(queue_);
      }
      current_tps_[current_index_].first = EnvTime::NowMicros(CLOCK_MONOTONIC); // 记录拷贝开始时间
      buffers->H2D(queue_);               // 设备到设备的数据拷贝操作
      current_tps_[current_index_].second = EnvTime::NowMicros(CLOCK_MONOTONIC); // 记录拷贝结束时间
      notifiers_[current_index_][1]->PlaceOn(queue_);
      // cpy on queue, so enqueue should wait for mlu queue
      out.dev_ = notifiers_[current_index_][1];  // 将设备时间通知器添加到输出Fifo中
    }
  }
  // 将处理完的数据推入输出Fifo中
  out_fifo_->push(out);
}

// 切换到下一个shape
void CpyIn::MoveShapeNext() {
  shape_idx_ = (shape_idx_ + 1) % all_shapes_.size();  // 循环切换到下一个shape
}

Enqueue::Enqueue(int enqueue_depth,
                 bool on_host,
                 bool dynamic_infer,
                 bool muta,
                 NotifierType t,
                 Fifo *cpy_in_fifo,
                 Fifo *enqueue_fifo,
                 BufferGroups *in_group,
                 BufferGroups *out_group,
                 IContext *context,
                 int dev_id,
                 uint64_t bind_bitmap,
                 TimeInfoContainer *tracer)
    : Stage(enqueue_depth, on_host, false, t, cpy_in_fifo, enqueue_fifo, in_group, out_group, tracer),
      context_(context) {
  if (dynamic_infer) {
    dyn_outs_.resize(enqueue_depth);
    function_ = [this](Buffers *in, Buffers *out, int idx) {
      notifiers_[idx][0]->PlaceOn(queue_);
      current_tps_[idx].first = EnvTime::NowMicros(CLOCK_MONOTONIC);
      CHECK_STATUS(context_->Enqueue(in->OriTensors(), &dyn_outs_[idx], queue_->Get()));
      current_tps_[idx].second = EnvTime::NowMicros(CLOCK_MONOTONIC);
      notifiers_[idx][1]->PlaceOn(queue_);
      out->Init(dyn_outs_[idx]);
    };
  } else if (muta) {
    function_ = [this](Buffers *in, Buffers *out, int idx) {
      notifiers_[idx][0]->PlaceOn(queue_);
      current_tps_[idx].first = EnvTime::NowMicros(CLOCK_MONOTONIC);
      CHECK_STATUS(context_->Enqueue(in->OriTensors(), out->OriTensors(), queue_->Get()));
      current_tps_[idx].second = EnvTime::NowMicros(CLOCK_MONOTONIC);
      notifiers_[idx][1]->PlaceOn(queue_);
      out->ReInit();
    };
  } else {
    function_ = [this](Buffers *in, Buffers *out, int idx) {
      notifiers_[idx][0]->PlaceOn(queue_);
      current_tps_[idx].first = EnvTime::NowMicros(CLOCK_MONOTONIC);
      CHECK_STATUS(context_->Enqueue(in->OriTensors(), out->OriTensors(), queue_->Get()));
      current_tps_[idx].second = EnvTime::NowMicros(CLOCK_MONOTONIC);
      notifiers_[idx][1]->PlaceOn(queue_);
    };
  }
  if (on_host_) {
    host_enqueue_ = [this](Buffers *in, Buffers *out, AtomicEvent *e, int idx) {
      if (e) {
        e->Wait();
      }
      function_(in, out, idx);
      events_[idx]->PlaceOn();
    };
    if (bind_bitmap > 0) {
      pool_->AddTask(BindCluster, dev_id, bind_bitmap);
    }
  }
  begin_ = new Notifier(!RecordHost(t_));
  SetDurationStart();
}

void Enqueue::CollectTime(int index) {
  last_duration_ = RecordHost(t_) ? notifiers_[index][0]->HostTimeFrom(*begin_)
                                  : notifiers_[index][0]->DevTimeFrom(*begin_);
  float interface_duration_ = float(current_tps_[index].second - current_tps_[index].first) / 1000;
  total_time_container_->emplace_back(
      RecordHost(t_) ? notifiers_[index][1]->HostTimeFrom(*notifiers_[index][0]) : 0,
      RecordDev(t_) ? notifiers_[index][1]->DevTimeFrom(*notifiers_[index][0]) : 0,
      interface_duration_);
}

void Enqueue::SetDurationStart() {
  begin_->PlaceOn(queue_);
  last_duration_ = 0;
}

void Enqueue::DoWork(const FifoUnit &unit) {
  FifoUnit out = unit;
  out.pipe_idxes_[1] = current_index_;
  auto in_buffers = (*in_group_)[out.pipe_idxes_[0]];
  auto out_buffers = (*out_group_)[current_index_];
  if (on_host_) {
    pool_->AddTask(host_enqueue_, in_buffers, out_buffers, out.host_, current_index_);
    out.host_ = events_[current_index_];
  } else {
    if (unit.dev_) {
      queue_->Wait(unit.dev_);
    }
    function_(in_buffers, out_buffers, current_index_);
    out.host_ = nullptr;
  }
  out.dev_ = notifiers_[current_index_][1];
  out_fifo_->push(out);
}

CpyOut::CpyOut(int enqueue_depth,
               bool on_host,
               bool do_cpy,
               NotifierType t,
               Fifo *enqueue_fifo,
               Fifo *cpy_out_fifo,
               BufferGroups *out_group,
               TimeInfoContainer *tracer)
    : Stage(enqueue_depth, on_host, !do_cpy, t, enqueue_fifo, cpy_out_fifo, nullptr, out_group, tracer) {
  if (on_host_) {
    host_cpy_ = [this](Buffers *buffer, AtomicEvent *env, Notifier *ntf, int idx) {
      env->Wait();
      ntf->Wait();
      current_tps_[idx].first = EnvTime::NowMicros(CLOCK_MONOTONIC);
      buffer->D2H();
      current_tps_[idx].second = EnvTime::NowMicros(CLOCK_MONOTONIC);
      events_[idx]->PlaceOn();
    };
  }
}

void CpyOut::CollectTime(int index) {
  // collect time
  if (!skip_) {
    float interface_duration_ =
        float(current_tps_[index].second - current_tps_[index].first) / 1000;
    if (on_host_) {
      total_time_container_->emplace_back(interface_duration_, interface_duration_,
                                          interface_duration_);
    } else {
      total_time_container_->emplace_back(
          RecordHost(t_) ? notifiers_[index][1]->HostTimeFrom(*notifiers_[index][0]) : 0,
          RecordDev(t_) ? notifiers_[index][1]->DevTimeFrom(*notifiers_[index][0]) : 0,
          interface_duration_);
    }
  }
}

void CpyOut::DoWork(const FifoUnit &unit) {
  FifoUnit out = unit;
  out.pipe_idxes_[2] = current_index_;
  auto buffers = (*out_group_)[out.pipe_idxes_[1]];
  if (!skip_) {
    if (on_host_) {
      pool_->AddTask(host_cpy_, buffers, unit.host_, unit.dev_, current_index_);
      out.host_ = events_[current_index_];
      out.dev_ = nullptr;
    } else {
      queue_->Wait(unit.dev_);
      if (t_ != NotifierType::none) {
        notifiers_[current_index_][0]->PlaceOn(queue_);
      }
      current_tps_[current_index_].first = EnvTime::NowMicros(CLOCK_MONOTONIC);
      buffers->D2H(queue_);
      current_tps_[current_index_].second = EnvTime::NowMicros(CLOCK_MONOTONIC);
      notifiers_[current_index_][1]->PlaceOn(queue_);
      out.dev_ = notifiers_[current_index_][1];
      out.host_ = nullptr;
    }
  }
  out_fifo_->push(out);
}
