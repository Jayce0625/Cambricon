#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_plugin_crop ./libmagicmind_plugin.so || (echo "Run sample plugin crop failed"; cd -; exit -1);
command ./cpu_CropAndResize || (echo "Run cpu crop and resize failed"; cd -; exit -1);
command mm_run --magicmind_model plugin_crop_sample_model --input_dims 4,4,1080,608 10,4 4 4 --iterations 1 --duration 0 --warmup 0 --input_files  ./input ./crop_params ./roi_nums ./pad_values || (echo "Run sample plugin crop failed"; cd -; exit -1);
cd -
