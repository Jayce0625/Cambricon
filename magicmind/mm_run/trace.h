/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Functions/Objects for trace inference time/power/etc.
 *************************************************************************/
#ifndef TRACE_H_
#define TRACE_H_
#include <array>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "common/timer.h"
#include "common/logger.h"
#include "common/device.h"
#include "common/json_util.h"
#include "mm_run/shape_groups.h"

/*
 * Enum and functions for notifier type.
 * Create notifier without host timestamp will reduce its record overhead.
 */
enum class NotifierType : uint8_t {
  none = 0x0,
  host = 0x1,
  dev  = 0x2,
  both = 0x3,
};

static const std::unordered_map<std::string, NotifierType> kNTStringTable = {
    {"none", NotifierType::none},
    {"host", NotifierType::host},
    {"dev", NotifierType::dev},
    {"both", NotifierType::both},
};

inline std::string NTypeToString(NotifierType t) {
  for (auto e : kNTStringTable) {
    if (e.second == t) {
      return e.first;
    }
  }
  return "invalid";
}

inline NotifierType StringToNType(const std::string &s) {
  for (auto e : kNTStringTable) {
    if (e.first == s) {
      return e.second;
    }
  }
  return NotifierType::none;
}

inline bool DoRecord(NotifierType t) {
  return static_cast<int>(t) > 0;
}

inline bool RecordHost(NotifierType t) {
  return static_cast<int>(t) % 2 > 0;
}

inline bool RecordDev(NotifierType t) {
  return static_cast<int>(t) > 1;
}
/*
 * Struct for statistic time duration (millisecond) from three sources:
 * 1. host clock from queue/async threads
 * 2. device clock from mlu (if action is on queue)
 * 3. host clock for just call interface
 * Now run uses three Containers for H2d/Compute/D2H
 */
struct TimeInfo {
  TimeInfo() {}
  TimeInfo(float host, float dev, float interface)
      : host_duration_(host), dev_duration_(dev), interface_duration_(interface) {}
  float host_duration_{0};
  float dev_duration_{0};
  float interface_duration_{0};
};
/*
 * Helper functions for get duration from time info
 */
float HostDuration(const TimeInfo &t);

float DeviceDuration(const TimeInfo &t);

float InterfaceDuration(const TimeInfo &t);

/*
 * Tool functions for time statistical
 */
TimeInfo operator+(const TimeInfo &a, const TimeInfo &b);

TimeInfo operator+=(TimeInfo &a, const TimeInfo &b);

using DeviceUtilInfoTrace = std::vector<DeviceUtilInfo>;
using PMUUtilInfoTrace    = std::vector<PMUCounter::PMUUtilInfo>;
using HostUtilInfoTrace   = std::vector<HostUtilInfo>;
using TimeInfoContainer   = std::vector<TimeInfo>;
/*
 * Struct for trace time in one thread.
 */
struct InferenceTrace {
  uint64_t host_start_{0};
  uint64_t host_end_{0};
  std::vector<int> input_indexs_{};
  std::array<TimeInfoContainer, 3> time_traces_;
};
/*
 * Trace data among all threads
 */
using InferenceTraceContainer = std::vector<InferenceTrace>;
/*
 * Return wall time in second
 */
float WallTime(const InferenceTraceContainer &c);

/*
 * Struct for printing performance results
 */
struct PerformanceResult {
  float min{0};
  float max{0};
  float mean{0};
  float median{0};
  float cv{0};
  float percentile90{0};
  float percentile95{0};
  float percentile99{0};
  json11::Json ToJson() const {
    std::map<std::string, json11::Json> objs;
    objs["min"]    = json11::Json(min);
    objs["max"]    = json11::Json(max);
    objs["mean"]   = json11::Json(mean);
    objs["median"] = json11::Json(median);
    objs["percentile|90%|95%|99%"] =
        GetJsonObjFromValue("", std::vector<float>({percentile90, percentile95, percentile99}));
    return json11::Json(objs);
  }
};

std::ostream &operator<<(std::ostream &out, const PerformanceResult &result);
/*
 * \brief Find percentile in a sorted vector of traces
 *
 * @param[in]
 * 1. trace_vec: vector of traces
 * 2. metric_getter: metric of trace in DT type
 */
template <typename T, typename DT>
float FindPercentile(float percentile,
                     const std::vector<T> &trace_vec,
                     std::function<DT(const T &)> metric_getter) {
  const int all     = static_cast<int>(trace_vec.size());
  const int exclude = static_cast<int>((1 - percentile / 100) * all);
  if (trace_vec.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  if (percentile < 0.0f || percentile > 100.0f) {
    SLOG(ERROR) << "percentile is not in [0, 100]!";
    return std::numeric_limits<float>::infinity();
  }
  return metric_getter(trace_vec[std::max(all - 1 - exclude, 0)]);
}

/*
 * \brief Find median in a sorted vector of traces
 *
 * @param[in]
 * 1. trace_vec: vector of traces
 * 2. metric_getter: metric of trace in DT type
 */
template <typename T, typename DT>
float FindMedian(const std::vector<T> &trace_vec, std::function<DT(const T &)> metric_getter) {
  if (trace_vec.empty()) {
    return std::numeric_limits<float>::infinity();
  }

  const int m = trace_vec.size() / 2;
  if (trace_vec.size() % 2) {
    return metric_getter(trace_vec[m]);
  }
  return (metric_getter(trace_vec[m - 1]) + metric_getter(trace_vec[m])) / 2;
}
/*
 * \brief Find mean in a sorted vector of traces
 *
 * @param[in]
 * 1. trace_vec: vector of traces
 * 2. metric_getter: metric of trace in DT type
 */
template <typename T, typename DT>
float FindMean(const std::vector<T> &trace_vec, std::function<DT(const T &)> metric_getter) {
  if (trace_vec.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  const auto metric_accumulator = [metric_getter](float acc, const T &a) {
    return acc + metric_getter(a);
  };
  return std::accumulate(trace_vec.begin(), trace_vec.end(), 0.0f, metric_accumulator) /
         trace_vec.size();
}
/*
 * \brief Find std in a sorted vector of traces
 *
 * @param[in]
 * 1. trace_vec: vector of traces
 * 2. metric_getter: metric of trace in DT type
 */
template <typename T, typename DT>
float FindCV(const std::vector<T> &trace_vec, std::function<DT(const T &)> metric_getter) {
  if (trace_vec.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  auto mean                     = FindMean(trace_vec, metric_getter);
  const auto metric_accumulator = [metric_getter, mean](float acc, const T &a) {
    return acc + (metric_getter(a) - mean) * (metric_getter(a) - mean);
  };
  return sqrtf(float(std::accumulate(trace_vec.begin(), trace_vec.end(), 0.0f, metric_accumulator) /
                     trace_vec.size())) /
         mean;
}
/*
 * \brief Get performance result from vector of traces by given metric.
 * @param [in]:
 * 1. trace_vec
 * 2. metric_getter
 *
 * @Note: T represents the type of trace and DT represents the datatype
 * of metric in the template
 * The input of metric_getter is a trace.
 */
template <typename T, typename DT>
PerformanceResult GetPerformanceResult(const std::vector<T> &trace_vec,
                                       std::function<DT(const T &)> metric_getter) {
  const auto metric_comp = [metric_getter](const T &a, const T &b) {
    return metric_getter(a) < metric_getter(b);
  };

  std::vector<T> vec_sort = trace_vec;
  std::sort(vec_sort.begin(), vec_sort.end(), metric_comp);
  PerformanceResult result;
  result.min    = metric_getter(vec_sort.front());
  result.max    = metric_getter(vec_sort.back());
  result.mean   = FindMean(vec_sort, metric_getter);
  result.median = FindMedian(vec_sort, metric_getter);
  // for analysis now, wont print out
  result.cv           = FindCV(vec_sort, metric_getter);
  result.percentile90 = FindPercentile(90, vec_sort, metric_getter);
  result.percentile95 = FindPercentile(95, vec_sort, metric_getter);
  result.percentile99 = FindPercentile(99, vec_sort, metric_getter);
  return result;
}

class Report {
 public:
  Report(const ShapeGroups &shapes,
         std::vector<int> dev_id,
         const std::vector<InferenceTraceContainer> &c,
         const std::vector<DeviceUtilInfoTrace> &dev_infos,
         const std::vector<PMUUtilInfoTrace> &pmu_infos,
         const HostUtilInfoTrace host_infos,
         NotifierType t,
         const std::vector<int> &avg_runs);
  void Print() const;
  void Analysis(bool host_async,
                bool mutable_in,
                bool mutable_out,
                int buf_depth,
                int infer_depth);
  std::vector<json11::Json> ToJsons() const;

 private:
  struct ReportPerDev {
    ReportPerDev(int id,
                 const ShapeGroups &shapes,
                 const InferenceTraceContainer &c,
                 const DeviceUtilInfoTrace &dev_info,
                 const PMUUtilInfoTrace &pmu_info,
                 NotifierType t,
                 const std::vector<int> &avg_runs);
    void Print() const;
    void PrintAnalysisData() const;
    void Analysis(bool host_async);
    json11::Json ToJson() const;
    int dev_id_{0};
    int thread_num_{1};
    float wall_time_{0};
    float throughput_{0};
    float iter_pers_{0};
    float total_compute_dev_{0};
    float total_compute_host_{0};
    size_t total_batch_sizes_{0};
    size_t total_iters_{0};
    size_t runs_per_avg_{0};
    // each input shape's h2d/compute/d2h/xhost/dev/interface + total latency
    std::vector<std::array<PerformanceResult, 11>> inputs_perf_;
    // average latency
    std::vector<std::array<PerformanceResult, 3>> avg_perf_;
    // dev
    std::vector<PerformanceResult> dev_util_;
    // pmu
    std::vector<PerformanceResult> pmu_util_;
    ShapeGroups shapes_;
    NotifierType t_;
    bool analysised_ = false;
    // analysis data
    // over head 1:interface time/host hwtime
    // over head 2:dev hwtime/host hwtime
    // in/enqueue/out
    std::vector<std::array<float, 3>> over_head1_;
    std::vector<std::array<float, 3>> over_head2_;
    // all cvs
    std::vector<std::array<float, 3>> interface_cvs_;
    std::vector<std::array<float, 3>> dt_cvs_;
    std::vector<std::array<float, 3>> ht_cvs_;
    // in:enq
    std::vector<float> cpin_enq_ratio_;
    // out:enq
    std::vector<float> cpout_enq_ratio_;
    float est_notifer_ratio_;
    float est_query_ratio_;
    std::vector<int> avg_runs_;
  };
  // cpu perf
  std::vector<PerformanceResult> cpu_util_;
  std::vector<ReportPerDev> reports_;
};
#endif  // TRACE_H_
