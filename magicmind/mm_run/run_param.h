/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#ifndef RUN_PARAM_H_
#define RUN_PARAM_H_
#include <vector>
#include <unordered_map>
#include <string>
#include "common/param.h"
#include "common/logger.h"

class RunParam : public ArgListBase {
  DECLARE_ARG(magicmind_model, (std::string))->SetDescription("Input MagicMind model.");
  DECLARE_ARG(input_dims, (std::vector<std::vector<int>>))
      ->SetDescription("Input shapes by order. '_' represents scalar.")
      ->SetDefault({});
  DECLARE_ARG(batch_size, (std::vector<int>))
      ->SetDescription(
          "Input batchsize by order, will override all highest dimensions for all inputs and not "
          "affect unrank/scalar inputs.")
      ->SetDefault({});
  DECLARE_ARG(run_config, (std::string))
      ->SetDescription("Config json file for groups of mutable shapes.")
      ->SetDefault({});
  DECLARE_ARG(input_files, (std::vector<std::string>))
      ->SetDescription("Real input files for one iteration of job.")
      ->SetDefault({});
  DECLARE_ARG(output_path, (std::string))
      ->SetDescription(
          "Real output files for one iteration of job. Only works when real inputs are given.")
      ->SetDefault({});
  DECLARE_ARG(plugin, (std::vector<std::string>))
      ->SetDescription("Plugin kernel libraries.")
      ->SetDefault({});
  DECLARE_ARG(devices, (std::vector<int>))
      ->SetDescription("MLU device ids for launch jobs.")
      ->SetDefault({"0"});
  DECLARE_ARG(threads, (int))
      ->SetDescription("Thread num for launch jobs, each thread will launch iteration jobs.")
      ->SetDefault({"1"});
  DECLARE_ARG(bind_cluster, (bool))
      ->SetDescription(
          "Enable cluster binding, each thread's task will bind on one certain cluster.")
      ->SetDefault({"false"});
  DECLARE_ARG(warmup, (int))
      ->SetDescription("Warmup time in ms before measuring performance")
      ->SetDefault({"200"});
  DECLARE_ARG(duration, (int))
      ->SetDescription("Run at least 'duration' ms wall clock time (after warmup).")
      ->SetDefault({"3000"});
  DECLARE_ARG(iterations, (int))
      ->SetDescription("Run at least 'iterations' times of launch each thread (after warmup).")
      ->SetDefault({"100"});
  DECLARE_ARG(disable_data_copy, (bool))
      ->SetDescription("To disable h2d&d2h copy and launch jobs with uninitialized inputs.")
      ->SetDefault({"false"});
  DECLARE_ARG(host_async, (bool))->SetDescription("Run in host async mode.")->SetDefault({"false"});
  DECLARE_ARG(buffer_depth, (int))
      ->SetDescription("I/O stream size optimization. MUST greater than 1.")
      ->SetDefault({"2"});
  DECLARE_ARG(infer_depth, (int))
      ->SetDescription("Enqueue stream size optimization. MUST greater than 1.")
      ->SetDefault({"2"});
  DECLARE_ARG(kernel_capture, (bool))
      ->SetDescription("Enable kernel capture.")
      ->SetDefault({"false"});
  DECLARE_ARG(trace_path, (std::string))
      ->SetDescription("Output performance trace to trach_path dir.")
      ->SetDefault({});
  DECLARE_ARG(debug_path, (std::string))
      ->SetDescription("Output intermedia tensor data to debug_path.")
      ->SetDefault({});
  DECLARE_ARG(perf_path, (std::string))
      ->SetDescription("Enable MagicMind profiler and output profiler data to perf_path.")
      ->SetDefault({});
  DECLARE_ARG(trace_pmu, (bool))
      ->SetDescription(
          "Enable pmu tracer for bandwidth occupancy. MUST be used in exclusive process.")
      ->SetDefault({"false"});
  DECLARE_ARG(trace_time, (std::string))
      ->SetDescription(
          "To choose which Notifier tracer to use for IO/Enqueue, will affect throughput.")
      ->SetAlternative({"none", "host", "dev", "both"})
      ->SetDefault({"host"});
  DECLARE_ARG(avg_runs, (std::vector<int>))
      ->SetDescription(
          "Two numbers. To print group num of average performance by following behaviour: "
          "num of runs/avg_run[0] > avg_run[1] ? print avg_run[1] : num of runs/avg_run[0] "
          "sets of average performance.")
      ->SetDefault({"100,10"});
};

#endif  // RUN_PARAM_H_
