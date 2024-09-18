#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_maxpool2d || (echo "Run sample avgpool2d failed"; cd -; exit -1);
command ./sample_runtime --model_path max_pool2d_model --input_dims 1,2,3,4 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model max_pool2d_model --iterations 1 --duration 0 --warmup 0 --input_dims 1,2,3,4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
