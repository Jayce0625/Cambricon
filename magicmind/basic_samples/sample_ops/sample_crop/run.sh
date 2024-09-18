#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_crop || (echo "Run sample crop failed"; cd -; exit -1);
command ./sample_runtime --model_path crop_model --input_dims 8,4,5,6 2,3,2,1 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model crop_model --iterations 1 --duration 0 --warmup 0 --input_dims 8,4,5,6 2,3,2,1 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
