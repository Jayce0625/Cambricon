#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
#static
# basic test
command ./mm_run --magicmind_model model3 --kernel_capture true --buffer_depth 4 --threads 2 --bind_cluster true || (echo "Statis test1 failed"; cd -; exit -1)
# 2,,3 means ignore middle batch size, since it's a scalar in network and has not batchsize at all
command ./mm_run --magicmind_model model2 --input_dims 1,2,3,4 _ 5,6,7 --batch_size 2,,3 --buffer_depth 4 --threads 2 --bind_cluster true --trace_time both || (echo "Statis test1 failed"; cd -; exit -1)
command ./mm_run --magicmind_model model2 --input_dims 1,2,3,4 _ 5,6,7 --batch_size 2,,3 --buffer_depth 4 --threads 2 --bind_cluster true --host_async true --trace_time both || (echo "Static test2 failed"; cd -; exit -1)
#dynamic
command ./mm_run --magicmind_model model2 --run_config example/shape_json1.json --buffer_depth 4 --threads 2 --bind_cluster true --trace_time none || (echo "Dynamic test1 failed"; cd -; exit -1)
command ./mm_run --magicmind_model model2 --run_config example/shape_json2.json --buffer_depth 4 --threads 2 --bind_cluster true --host_async true --trace_time dev || (echo "Dynamic test2 failed"; cd -; exit -1)
# tool test
command ./mm_run --magicmind_model model2 --input_dims 1,2,3,4 _ 5,6,7 --warmup 0 --iterations 1 --duration 0 --input_files test/input1,test/input1,test/input1 --output_path ./ --trace_path ./ --debug_path ./ --perf_path ./ || (echo "Basic test failed"; cd -; exit -1)
pip3 install -r ../tools/perf_reader/requirements.txt
command ../tools/perf_reader/read_tensorboard --perf_path ./plugins || (echo "Read perf failed"; cd -; exit -1)
cd -
