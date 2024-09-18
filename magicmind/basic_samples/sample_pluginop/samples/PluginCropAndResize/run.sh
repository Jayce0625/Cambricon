#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_CropAndResize ./libmagicmind_plugin.so || (echo "Run sample crop and resize failed"; cd -; exit -1);
command ./cpu_CropAndResize || (echo "Run cpu crop and resize failed"; cd -; exit -1);
command ./sample_runtime --model_path crop_and_resize_model --input_dims 4,4,1080,608 10,4 4 4 --plugin_libs ./libmagicmind_plugin.so --data_path ./input ./crop_params ./roi_nums ./pad_values || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model crop_and_resize_model --input_dims 4,4,1080,608 10,4 4 4 --iterations 1 --duration 0 --warmup 0 --plugin ./libmagicmind_plugin.so --input_files ./input ./crop_params ./roi_nums ./pad_values || (echo "Run crop_and_resize failed"; cd -; exit -1);
command diff_compare --data output0 --baseline baseline --datatype uint8 --threshold1 0.005 --threshold2 0.005|| (echo "Run compare failed"; cd -; exit -1);
cd -
