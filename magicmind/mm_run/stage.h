/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Objects to describe stages among inference progress
 *************************************************************************/
#ifndef STAGE_H_
#define STAGE_H_
#include <vector>
#include <array>
#include <string>
#include <functional>
#include "mm_runtime.h"
#include "common/device.h"
#include "common/container.h"
#include "common/buffer.h"
#include "common/threadpool.h"
#include "mm_run/trace.h"
#include "mm_run/shape_groups.h"

using namespace magicmind;
// Reference to groups of ins/outs
using BufferGroups = std::vector<Buffers *>;
/*
 * A struct to decribe data between stages. A complete inference progress will pass
 * units from cpyin to enqueue, from enqueue to cpyout, and do sync at last stage.
 * pipe_idxes_ means which cpyin/enqueue/cpyout are occupied by this inference progress.
 * dev/host are the sync signal between async stages.
 * shape_idx means which group of shapes is using for this inference progress
 */
struct FifoUnit {
  std::array<int, 3> pipe_idxes_ = {{-1, -1, -1}};
  Notifier *dev_                 = nullptr;
  AtomicEvent *host_             = nullptr;
  int shape_idx_                 = 0;
};

using Fifo = RingQueue<FifoUnit>;

/*
 * A common abstract object for cpyin/out/enqueue.
 * Stage will be inited by its work depth, its device (to use a thread or a dev queue to perform
 * work) and two fifos for its input and output. Stage will collect its index'th time info when
 * index'th job is considered finished.
 */
class Stage {
 public:
  Stage(int depth,
        bool on_host,
        bool skip,
        NotifierType t,
        Fifo *in_fifo,
        Fifo *out_fifo,
        BufferGroups *in_group,
        BufferGroups *out_group,
        TimeInfoContainer *tracer);
  bool DoStage();
  void ResetActive(int index);
  virtual void DoWork(const FifoUnit &unit) = 0;
  virtual void CollectTime(int idx)         = 0;
  virtual ~Stage();

 private:
  void MoveNext();

 protected:
  Fifo *in_fifo_;
  Fifo *out_fifo_;
  bool on_host_ = false;
  bool skip_    = false;
  NotifierType t_;
  int depth_;
  // on dev
  Queue *queue_ = nullptr;
  std::vector<std::array<Notifier *, 2>> notifiers_;
  // on host
  ThreadPool *pool_ = nullptr;
  std::vector<AtomicEvent *> events_;
  BufferGroups *in_group_;
  BufferGroups *out_group_;
  TimeInfoContainer *total_time_container_;
  std::vector<std::pair<uint64_t, uint64_t>> current_tps_;
  std::vector<bool> active_;
  int current_index_ = 0;
};
/*
 * All stages doesnot hold any resources!
 */
class CpyIn : public Stage {
 public:
  CpyIn(int buffer_depth,
        bool on_host,
        bool do_cpy,
        NotifierType t,
        Fifo *cpy_in_fifo,
        BufferGroups *in_group,
        const std::vector<std::vector<Dims>> &all_shapes,
        TimeInfoContainer *tracer);
  void DoWork(const FifoUnit &unit) override final;
  void CollectTime(int idx) override final;

 private:
  void MoveShapeNext();

 private:
  std::vector<std::vector<Dims>> all_shapes_;
  int shape_idx_ = 0;
  std::function<void(Buffers *, int)> host_cpy_;
};

class Enqueue : public Stage {
 public:
  Enqueue(int enqueue_depth,
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
          TimeInfoContainer *tracer);
  ~Enqueue() {
    delete begin_;
    for (auto e_ : dyn_outs_) {
      for (auto t_ : e_) {
        t_->Destroy();
      }
    }
  }
  void SetDurationStart();
  void DoWork(const FifoUnit &unit) override final;
  void CollectTime(int idx) override final;
  float LastDuration() const { return last_duration_; }

 private:
  Notifier *begin_;
  float last_duration_ = 0;
  std::vector<std::vector<IRTTensor *>> dyn_outs_;
  IContext *context_ = nullptr;
  std::function<void(Buffers *, Buffers *, int)> function_;
  std::function<void(Buffers *, Buffers *, AtomicEvent *, int)> host_enqueue_;
};

class CpyOut : public Stage {
 public:
  CpyOut(int buffer_depth,
         bool on_host,
         bool do_cpy,
         NotifierType t,
         Fifo *enqueue_fifo,
         Fifo *cpy_out_fifo,
         BufferGroups *out_group,
         TimeInfoContainer *tracer);
  void DoWork(const FifoUnit &unit) override final;
  void CollectTime(int idx) override final;

 private:
  std::function<void(Buffers *, AtomicEvent *, Notifier *, int)> host_cpy_;
};

#endif  // STAGE_H_
