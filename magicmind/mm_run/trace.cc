/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Functions/Objects for trace inference time/power/etc.
 *************************************************************************/
#include <iomanip>
#include "mm_run/trace.h"

TimeInfo operator+(const TimeInfo &a, const TimeInfo &b) {
  return TimeInfo(a.host_duration_ + b.host_duration_, a.dev_duration_ + b.dev_duration_,
                  a.interface_duration_ + b.interface_duration_);
}

TimeInfo operator+=(TimeInfo &a, const TimeInfo &b) {
  return a = a + b;
}

float HostDuration(const TimeInfo &t) {
  return t.host_duration_;
}

float DeviceDuration(const TimeInfo &t) {
  return t.dev_duration_;
}

float InterfaceDuration(const TimeInfo &t) {
  return t.interface_duration_;
}

float WallTime(const std::vector<InferenceTrace> &c) {
  if (c.size() == 0) {
    return 0;
  }
  uint64_t end = c.back().host_end_;
  uint64_t start = c.front().host_start_;
  for (auto trace : c) {
    start = trace.host_start_ < start ? trace.host_start_ : start;
    end = trace.host_end_ > end ? trace.host_end_ : end;
  }
  // micro to second
  return float(end - start) / (1000 * 1000);
}

std::ostream &operator<<(std::ostream &out, const PerformanceResult &r) {
  out << "min: " << std::setw(10) << std::left << std::setprecision(5) << r.min;
  out << " max: " << std::setw(10) << std::left << std::setprecision(5) << r.max;
  out << " mean: " << std::setw(10) << std::left << std::setprecision(5) << r.mean;
  out << " median: " << std::setw(10) << std::left << std::setprecision(5) << r.median;
  out << " percentile:";
  out << " (90%) " << std::setw(10) << std::left << std::setprecision(5) << r.percentile90;
  out << " (95%) " << std::setw(10) << std::left << std::setprecision(5) << r.percentile95;
  out << " (99%) " << std::setw(10) << std::left << std::setprecision(5) << r.percentile99;
  return out;
}

namespace {
size_t TotalBatchSize(const std::vector<int> &batch_sizes,
                      const std::vector<int> &input_index_gather) {
  size_t total_size = 0;
  for (auto e_ : input_index_gather) {
    total_size += batch_sizes[e_];
  }
  return total_size;
}

std::vector<std::array<TimeInfoContainer, 4>> CateTimeInfo(
    const std::vector<int> &input_index_gather,
    const std::array<TimeInfoContainer, 4> &c) {
  std::vector<std::array<TimeInfoContainer, 4>> ret;
  int max_idx = 0;
  for (size_t idx = 0; idx < input_index_gather.size(); ++idx) {
    max_idx = max_idx > input_index_gather[idx] ? max_idx : input_index_gather[idx];
  }
  ret.resize(max_idx + 1);
  for (size_t idx = 0; idx < input_index_gather.size(); ++idx) {
    for (int arr = 0; arr < 4; ++arr) {
      ret[input_index_gather[idx]][arr].push_back(c[arr][idx]);
    }
  }
  return ret;
}
}  // namespace

Report::Report(const ShapeGroups &shapes,
               std::vector<int> dev_id,
               const std::vector<InferenceTraceContainer> &c,
               const std::vector<DeviceUtilInfoTrace> &dev_infos,
               const std::vector<PMUUtilInfoTrace> &pmu_infos,
               const HostUtilInfoTrace host_infos,
               NotifierType t,
               const std::vector<int> &avg_runs) {
  CHECK_EQ(avg_runs.size(), 2);
  for (size_t i = 0; i < dev_id.size(); ++i) {
    reports_.emplace_back(dev_id[i], shapes, c[i], dev_infos[i], pmu_infos[i], t, avg_runs);
  }
  if (!host_infos.empty()) {
    cpu_util_.push_back(GetPerformanceResult<HostUtilInfo, double>(host_infos, UserOcc));
    cpu_util_.push_back(GetPerformanceResult<HostUtilInfo, double>(host_infos, KernelOcc));
    cpu_util_.push_back(GetPerformanceResult<HostUtilInfo, double>(host_infos, VmUsage));
    cpu_util_.push_back(GetPerformanceResult<HostUtilInfo, double>(host_infos, ResUsage));
  }
}

Report::ReportPerDev::ReportPerDev(int id,
                                   const ShapeGroups &shapes,
                                   const InferenceTraceContainer &c,
                                   const DeviceUtilInfoTrace &dev_info,
                                   const PMUUtilInfoTrace &pmu_info,
                                   NotifierType t,
                                   const std::vector<int> &avg_runs)
    : dev_id_(id), shapes_(shapes), t_(t), avg_runs_(avg_runs) {
  /////////////////////////////////////Main perf data//////////////////////////////////////
  if (c.size() == 0) {
    SLOG(ERROR) << "Empty trace data.";
    abort();
  }
  wall_time_ = WallTime(c);
  thread_num_ = c.size();
  // h2d/enqueue/d2h x host async/dev async/host interface clocks
  // so there are 9 time traces totally
  std::vector<int> input_index_gather;
  std::array<TimeInfoContainer, 4> time_trace_gather;
  for (auto &trace : c) {
    time_trace_gather[0].insert(time_trace_gather[0].end(), trace.time_traces_[0].begin(),
                                trace.time_traces_[0].end());
    time_trace_gather[1].insert(time_trace_gather[1].end(), trace.time_traces_[1].begin(),
                                trace.time_traces_[1].end());
    time_trace_gather[2].insert(time_trace_gather[2].end(), trace.time_traces_[2].begin(),
                                trace.time_traces_[2].end());
    input_index_gather.insert(input_index_gather.end(), trace.input_indexs_.begin(),
                              trace.input_indexs_.end());
  }
  total_iters_ = time_trace_gather[1].size();
  if (time_trace_gather[0].size() == 0) {
    time_trace_gather[0].resize(total_iters_);
  }
  if (time_trace_gather[2].size() == 0) {
    time_trace_gather[2].resize(total_iters_);
  }
  CHECK_EQ(time_trace_gather[0].size(), time_trace_gather[1].size());
  CHECK_EQ(time_trace_gather[1].size(), time_trace_gather[2].size());

  for (auto t : time_trace_gather[1]) {
    total_compute_dev_ += t.dev_duration_;
    total_compute_host_ += t.host_duration_;
  }

  time_trace_gather[3] = TimeInfoContainer(total_iters_);
  for (size_t idx = 0; idx < total_iters_; ++idx) {
    time_trace_gather[3][idx] += time_trace_gather[0][idx];
    time_trace_gather[3][idx] += time_trace_gather[1][idx];
    time_trace_gather[3][idx] += time_trace_gather[2][idx];
  }

  auto batch_sizes = shapes.BatchSizes();
  total_batch_sizes_ = TotalBatchSize(batch_sizes, input_index_gather);
  throughput_ = total_batch_sizes_ / wall_time_;
  iter_pers_ = total_iters_ / wall_time_;
  auto cated_info = CateTimeInfo(input_index_gather, time_trace_gather);
  inputs_perf_.resize(shapes_.size());
  for (size_t idx = 0; idx < shapes_.size(); ++idx) {
    // H2D
    inputs_perf_[idx][0] = GetPerformanceResult<TimeInfo, float>(cated_info[idx][0], HostDuration);
    inputs_perf_[idx][1] =
        GetPerformanceResult<TimeInfo, float>(cated_info[idx][0], DeviceDuration);
    inputs_perf_[idx][2] =
        GetPerformanceResult<TimeInfo, float>(cated_info[idx][0], InterfaceDuration);
    // Enqueue
    inputs_perf_[idx][3] = GetPerformanceResult<TimeInfo, float>(cated_info[idx][1], HostDuration);
    inputs_perf_[idx][4] =
        GetPerformanceResult<TimeInfo, float>(cated_info[idx][1], DeviceDuration);
    inputs_perf_[idx][5] =
        GetPerformanceResult<TimeInfo, float>(cated_info[idx][1], InterfaceDuration);
    // D2H
    inputs_perf_[idx][6] = GetPerformanceResult<TimeInfo, float>(cated_info[idx][2], HostDuration);
    inputs_perf_[idx][7] =
        GetPerformanceResult<TimeInfo, float>(cated_info[idx][2], DeviceDuration);
    inputs_perf_[idx][8] =
        GetPerformanceResult<TimeInfo, float>(cated_info[idx][2], InterfaceDuration);
    // latency
    inputs_perf_[idx][9] = GetPerformanceResult<TimeInfo, float>(cated_info[idx][3], HostDuration);
    inputs_perf_[idx][10] =
        GetPerformanceResult<TimeInfo, float>(cated_info[idx][3], DeviceDuration);
  }
  // average infos are only useful on actual time line
  // so for multithreads avg performance, we use the same range in each thread's performance as
  // valid data
  int avg_group_num_ = int(total_iters_ / avg_runs_[0]) > avg_runs_[1] ? avg_runs_[1] : total_iters_ / avg_runs_[0];
  if (avg_group_num_ > 0) {
    runs_per_avg_ = total_iters_ / avg_group_num_;
    for (int g = 0; g < avg_group_num_; ++g) {
      TimeInfoContainer avg;  // only consider enqueue time
      for (auto t : c) {
        size_t range = t.time_traces_[1].size() / avg_group_num_;
        size_t begin = range * g;
        avg.insert(avg.end(), t.time_traces_[1].begin() + begin,
                   t.time_traces_[1].begin() + begin + range);
      }
      std::array<PerformanceResult, 3> perf;
      perf[0] = GetPerformanceResult<TimeInfo, float>(avg, HostDuration);
      perf[1] = GetPerformanceResult<TimeInfo, float>(avg, DeviceDuration);
      perf[2] = GetPerformanceResult<TimeInfo, float>(avg, InterfaceDuration);
      avg_perf_.push_back(perf);
    }
  }
  //////////////////////////////////////Dev utils///////////////////////////////////////
  if (!dev_info.empty()) {
    dev_util_.push_back(GetPerformanceResult<DeviceUtilInfo, double>(dev_info, CoreUtil));
    dev_util_.push_back(GetPerformanceResult<DeviceUtilInfo, double>(dev_info, MemUtil));
    dev_util_.push_back(GetPerformanceResult<DeviceUtilInfo, double>(dev_info, PowerUtil));
    dev_util_.push_back(GetPerformanceResult<DeviceUtilInfo, double>(dev_info, TempUtil));
  }
  //////////////////////////////////////PMU utils///////////////////////////////////////
  if (!pmu_info.empty()) {
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, DRAMRead));
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, DRAMWrite));
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, PCIERead));
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, PCIEWrite));
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, CoreRead));
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, CoreWrite));
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, ALUCycles));
    pmu_util_.push_back(GetPerformanceResult<PMUCounter::PMUUtilInfo, double>(pmu_info, LTCycles));
  }

}

void Report::ReportPerDev::Print() const {
  std::stringstream os;
  if (!dev_util_.empty()) {
    os << "\n=================== Device " << dev_id_ << " Utilization Summary\n";
    os << std::setw(30) << std::left << "chip_util(%): " << dev_util_[0] << std::endl
       << std::setw(30) << std::left << "mem_util(MB): " << dev_util_[1] << std::endl
       << std::setw(30) << std::left << "power_util(W): " << dev_util_[2] << std::endl
       << std::setw(30) << std::left << "temp_util(C): " << dev_util_[3];
  }
  if (!pmu_util_.empty()) {
    os << "\n=================== Device " << dev_id_ << " PMU Summary\n";
    os << std::setw(30) << std::left << "DRAM read(MB/s): " << pmu_util_[0] << std::endl
       << std::setw(30) << std::left << "DRAM write(MB/s): " << pmu_util_[1] << std::endl
       << std::setw(30) << std::left << "PCIE read(MB/s): " << pmu_util_[2] << std::endl
       << std::setw(30) << std::left << "PCIE write(MB/s): " << pmu_util_[3] << std::endl
       << std::setw(30) << std::left << "core read(MB/s): " << pmu_util_[4] << std::endl
       << std::setw(30) << std::left << "core write(MB/s): " << pmu_util_[5] << std::endl
       << std::setw(30) << std::left << "alu cycles(cycles/us): " << pmu_util_[6] << std::endl
       << std::setw(30) << std::left << "lt cycles(cycles/us): " << pmu_util_[7];
  }
  os << "\n=================== Performance Summary\n";
  os << std::setw(40) << std::left << "Total iterations: " << total_iters_ << std::endl;
  os << std::setw(40) << std::left << "Host wall time(s): " << wall_time_ << std::endl;
  if (RecordHost(t_)) {
    os << std::setw(40) << std::left
       << "Total MLU compute time(host clock, s): " << total_compute_host_ / 1000 << std::endl;
  }
  if (RecordDev(t_)) {
    os << std::setw(40) << std::left
       << "Total MLU compute time(dev clock, s): " << total_compute_dev_ / 1000 << std::endl;
  }
  os << std::setw(40) << std::left << "Throughput (qps): " << throughput_
     << " with total batch sizes: " << total_batch_sizes_ << " and " << iter_pers_
     << " iterations/second" << std::endl;
  for (size_t shape_idx = 0; shape_idx < shapes_.size(); ++shape_idx) {
    os << std::setw(40) << "Input shape group " << shape_idx << ": "
       << shapes_[shape_idx].DebugString() << std::endl;
    os << "  H2D Summary:\n";
    if (RecordHost(t_)) {
      os << std::setw(30) << std::left
         << "  Latency(host clock, ms): " << inputs_perf_[shape_idx][0] << std::endl;
    }
    if (RecordDev(t_)) {
      os << std::setw(30) << std::left << "  Latency(dev clock, ms): " << inputs_perf_[shape_idx][1]
         << std::endl;
    }
    os << std::setw(30) << std::left << "  Interface Duration(ms): " << inputs_perf_[shape_idx][2]
       << std::endl;
    os << "  MLU Compute (Launch jobs) Summary:\n";
    if (RecordHost(t_)) {
      os << std::setw(30) << std::left
         << "  Latency(host clock, ms): " << inputs_perf_[shape_idx][3] << std::endl;
    }
    if (RecordDev(t_)) {
      os << std::setw(30) << std::left << "  Latency(dev clock, ms): " << inputs_perf_[shape_idx][4]
         << std::endl;
    }
    os << std::setw(30) << std::left << "  Interface Duration(ms): " << inputs_perf_[shape_idx][5]
       << std::endl;
    os << "  D2H Summary:\n";
    if (RecordHost(t_)) {
      os << std::setw(30) << std::left
         << "  Latency(host clock, ms): " << inputs_perf_[shape_idx][6] << std::endl;
    }
    if (RecordDev(t_)) {
      os << std::setw(30) << std::left << "  Latency(dev clock, ms): " << inputs_perf_[shape_idx][7]
         << std::endl;
    }
    os << std::setw(30) << std::left << "  Interface Duration(ms): " << inputs_perf_[shape_idx][8]
       << std::endl;
    if (DoRecord(t_)) {
      os << "  Total Latency Summary(H2D + Compute + D2H):\n";
    }
    if (RecordHost(t_)) {
      os << std::setw(30) << std::left
         << "  Latency(host clock, ms): " << inputs_perf_[shape_idx][9] << std::endl;
    }
    if (RecordDev(t_)) {
      os << std::setw(30) << std::left
         << "  Latency(dev clock, ms): " << inputs_perf_[shape_idx][10] << std::endl;
    }
  }
  if (avg_perf_.size()) {
    os << "Trace average MLU Compute perf over " << runs_per_avg_ << ":" << std::endl;
    for (auto perf : avg_perf_) {
      if (RecordHost(t_)) {
        os << "  Latency(host clock, ms): " << std::setw(10) << std::left << std::setprecision(5)
           << perf[0].mean;
      }
      if (RecordDev(t_)) {
        os << "  Latency(dev clock, ms): " << std::setw(10) << std::left << std::setprecision(5)
           << perf[1].mean;
      }
      os << "  Interface Duration(ms): " << std::setw(10) << std::left << std::setprecision(5)
         << perf[2].mean << std::endl;
    }
  }
  SLOG(INFO) << os.str();
}

void Report::ReportPerDev::Analysis(bool host_async) {
  over_head1_.resize(shapes_.size());
  over_head2_.resize(shapes_.size());
  interface_cvs_.resize(shapes_.size());
  dt_cvs_.resize(shapes_.size());
  ht_cvs_.resize(shapes_.size());
  cpin_enq_ratio_.resize(shapes_.size());
  cpout_enq_ratio_.resize(shapes_.size());
  constexpr float InterfaceThres {0.8f};
  est_notifer_ratio_ = iter_pers_ * 0.4 * 1e-3; // 2 waits, 2*20us
  if (host_async) {
    est_notifer_ratio_ = est_notifer_ratio_ / 2; // 1 wait, 1*20us
  }
  est_query_ratio_ = iter_pers_ * 1.4 * 1e-3; // 7 querys, 2*20us
  if (host_async) {
    est_query_ratio_ = est_notifer_ratio_ / 7 * 5; // 5 querys, 1*20us
  }
  for (size_t idx = 0; idx < shapes_.size(); ++idx) {
    over_head1_[idx][0] = (inputs_perf_[idx][2].median / InterfaceThres / inputs_perf_[idx][0].median - 1) * 100;
    over_head1_[idx][1] = (inputs_perf_[idx][5].median / InterfaceThres / inputs_perf_[idx][3].median - 1) * 100;
    over_head1_[idx][2] = (inputs_perf_[idx][8].median / InterfaceThres / inputs_perf_[idx][6].median - 1) * 100;
    if (over_head1_[idx][0] < 0 || host_async) over_head1_[idx][0] = 0;
    if (over_head1_[idx][1] < 0) over_head1_[idx][1] = 0;
    if (over_head1_[idx][2] < 0 || host_async) over_head1_[idx][2] = 0;
    over_head2_[idx][0] = (1 - inputs_perf_[idx][1].median / inputs_perf_[idx][0].median) * 100;
    over_head2_[idx][1] = (1 - inputs_perf_[idx][4].median / inputs_perf_[idx][3].median) * 100;
    over_head2_[idx][2] = (1 - inputs_perf_[idx][7].median / inputs_perf_[idx][6].median) * 100;
    interface_cvs_[idx][0] = inputs_perf_[idx][2].cv * 100;
    interface_cvs_[idx][1] = inputs_perf_[idx][5].cv * 100;
    interface_cvs_[idx][2] = inputs_perf_[idx][8].cv * 100;
    dt_cvs_[idx][0] = inputs_perf_[idx][1].cv * 100;
    dt_cvs_[idx][1] = inputs_perf_[idx][4].cv * 100;
    dt_cvs_[idx][2] = inputs_perf_[idx][7].cv * 100;
    ht_cvs_[idx][0] = inputs_perf_[idx][0].cv * 100;
    ht_cvs_[idx][1] = inputs_perf_[idx][3].cv * 100;
    ht_cvs_[idx][2] = inputs_perf_[idx][6].cv * 100;
    cpin_enq_ratio_[idx] = inputs_perf_[idx][0].median / inputs_perf_[idx][3].median * 100;
    cpout_enq_ratio_[idx] = inputs_perf_[idx][6].median / inputs_perf_[idx][3].median * 100;
  }
  analysised_ = true;
}

void Report::ReportPerDev::PrintAnalysisData() const {
  std::stringstream os;
  auto print = [](const std::array<float, 3> &in) {
    std::stringstream os;
    os << "H2D: " << std::setw(12) << std::left << in[0];
    os << "Compute: " << std::setw(12) << std::left << in[1];
    os << "D2H: " << std::setw(12) << std::left << in[2];
    return os.str();
  };
  os << "\n=================== Report Dev " << dev_id_ << " Analysis Summary\n";
  // dev/host threadhold
  os << std::setw(30) << std::left << "Estimate notifier ratio(%): " << est_notifer_ratio_
     << std::endl;
  os << std::setw(30) << std::left << "Estimate query ratio(%): " << est_query_ratio_
     << std::endl;
  for (size_t idx = 0; idx < shapes_.size(); ++idx) {
    os << std::setw(30) << "Input shape group " << idx << ": " << shapes_[idx].DebugString()
       << std::endl;
    os << std::setw(30) << std::left << "OverHead1(%): " << print(over_head1_[idx]) << std::endl;
    os << std::setw(30) << std::left << "OverHead2(%): " << print(over_head2_[idx]) << std::endl;
    os << std::setw(30) << std::left << "Interface C.V(%): " << print(interface_cvs_[idx])
       << std::endl;
    os << std::setw(30) << std::left << "Dev Time C.V(%): " << print(dt_cvs_[idx]) << std::endl;
    os << std::setw(30) << std::left << "Host Time C.V(%): " << print(ht_cvs_[idx]) << std::endl;
    os << std::setw(30) << std::left << "H2D/Compute ratio(%): " << cpin_enq_ratio_[idx]
       << std::endl;
    os << std::setw(30) << std::left << "D2H/Compute ratio(%): " << cpout_enq_ratio_[idx]
       << std::endl;
  }
  SLOG(INFO) << os.str();
}

json11::Json Report::ReportPerDev::ToJson() const {
  std::map<std::string, json11::Json> objs;
  objs["iterations"] = json11::Json((int)total_iters_);
  objs["inputType"] = json11::Json(shapes_.has_name() ? 1 : 0);
  objs["mluComputTime(s)"] = json11::Json(total_compute_host_);
  objs["mluComputTime(dev|s)"] = json11::Json(total_compute_dev_);
  if (!dev_util_.empty()) {
    objs["chip_util(%)"] = dev_util_[0].ToJson();
    objs["mem_util(MB)"] = dev_util_[1].ToJson();
    objs["power_util(W)"] = dev_util_[2].ToJson();
    objs["temp_util(C)"] = dev_util_[3].ToJson();
  }
  if (!pmu_util_.empty()) {
    objs["DRAM read(MB/s)"] = pmu_util_[0].ToJson();
    objs["DRAM write(MB/s)"] = pmu_util_[1].ToJson();
    objs["PCIE read(MB/s)"] = pmu_util_[2].ToJson();
    objs["PCIE write(MB/s)"] = pmu_util_[3].ToJson();
    objs["core read(MB/s)"] = pmu_util_[4].ToJson();
    objs["core write(MB/s)"] = pmu_util_[5].ToJson();
    objs["alu cycles(cycles/us)"] = pmu_util_[6].ToJson();
    objs["lt cycles(cycles/us)"] = pmu_util_[7].ToJson();
  }
  if (analysised_) {
    objs["notifier ratio(%)"] = json11::Json(est_notifer_ratio_);
  }
  std::vector<json11::Json> traces;
  for (size_t idx = 0; idx < shapes_.size(); idx++) {
    std::map<std::string, json11::Json> trace;
    trace["inputDims"] = shapes_[idx].ToJson();
    if (RecordHost(t_)) {
      trace["h2d(ms)"] = inputs_perf_[idx][0].ToJson();
    }
    if (RecordDev(t_)) {
      trace["h2d(dev|ms)"] = inputs_perf_[idx][1].ToJson();
    }
    trace["h2d interface(ms)"] = inputs_perf_[idx][2].ToJson();
    if (RecordHost(t_)) {
      trace["mlu(ms)"] = inputs_perf_[idx][3].ToJson();
    }
    if (RecordDev(t_)) {
      trace["mlu(dev|ms)"] = inputs_perf_[idx][4].ToJson();
    }
    trace["enqueue(ms)"] = inputs_perf_[idx][5].ToJson();
    if (RecordHost(t_)) {
      trace["d2h(ms)"] = inputs_perf_[idx][6].ToJson();
    }
    if (RecordDev(t_)) {
      trace["d2h(dev|ms)"] = inputs_perf_[idx][7].ToJson();
    }
    trace["d2h interface(ms)"] = inputs_perf_[idx][8].ToJson();
    if (RecordHost(t_)) {
      trace["latency(host|ms)"] = inputs_perf_[idx][9].ToJson();
    }
    if (RecordDev(t_)) {
      trace["latency(dev|ms)"] = inputs_perf_[idx][10].ToJson();
    }
    if (analysised_) {
      std::map<std::string, json11::Json> overhead1;
      overhead1["H2D"] = json11::Json(over_head1_[idx][0]);
      overhead1["Enq"] = json11::Json(over_head1_[idx][1]);
      overhead1["D2H"] = json11::Json(over_head1_[idx][2]);
      std::map<std::string, json11::Json> overhead2;
      overhead2["H2D"] = json11::Json(over_head2_[idx][0]);
      overhead2["Enq"] = json11::Json(over_head2_[idx][1]);
      overhead2["D2H"] = json11::Json(over_head2_[idx][2]);
      std::map<std::string, json11::Json> inter_cv;
      inter_cv["H2D"] = json11::Json(interface_cvs_[idx][0]);
      inter_cv["Enq"] = json11::Json(interface_cvs_[idx][1]);
      inter_cv["D2H"] = json11::Json(interface_cvs_[idx][2]);
      std::map<std::string, json11::Json> dev_cv;
      dev_cv["H2D"] = json11::Json(dt_cvs_[idx][0]);
      dev_cv["Enq"] = json11::Json(dt_cvs_[idx][1]);
      dev_cv["D2H"] = json11::Json(dt_cvs_[idx][2]);
      std::map<std::string, json11::Json> host_cv;
      host_cv["H2D"] = json11::Json(ht_cvs_[idx][0]);
      host_cv["Enq"] = json11::Json(ht_cvs_[idx][1]);
      host_cv["D2H"] = json11::Json(ht_cvs_[idx][2]);
      trace["OverHead1(%)"] = json11::Json(overhead1);
      trace["OverHead2(%)"] = json11::Json(overhead2);
      trace["Interface CV(%)"] = json11::Json(inter_cv);
      trace["Dev CV(%)"] = json11::Json(dev_cv);
      trace["Host CV(%)"] = json11::Json(host_cv);
    }
    traces.push_back(json11::Json(trace));
  }
  objs["trace"] = json11::Json(traces);
  return json11::Json(objs);
}

void Report::Print() const {
  std::stringstream os;
  if (!cpu_util_.empty()) {
    os << "\n=================== Host Occupancy Summary\n";
    os << std::setw(30) << std::left << "UserMode Occupancy(%): " << cpu_util_[0] << std::endl
       << std::setw(30) << std::left << "KernelMode Occupancy(%): " << cpu_util_[1] << std::endl
       << std::setw(30) << std::left << "VirtualMem Usage(GB): " << cpu_util_[2] << std::endl
       << std::setw(30) << std::left << "ResidentSet Usage(GB): " << cpu_util_[3];
    SLOG(INFO) << os.str();
  }
  for (auto r : reports_) {
    r.Print();
  }
}

void Report::Analysis(bool host_async,
                      bool mutable_in,
                      bool mutable_out,
                      int buf_depth,
                      int infer_depth) {
  for (auto &r : reports_) {
    if (r.t_ == NotifierType::both) {
      r.Analysis(host_async);
      r.PrintAnalysisData();
    }
  }
}

std::vector<json11::Json> Report::ToJsons() const {
  std::vector<json11::Json> ret;
  for (auto r : reports_) {
    ret.push_back(r.ToJson());
  }
  return ret;
}
