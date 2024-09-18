#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_resize || (echo "Run sample resize failed"; cd -; exit -1);
command ./sample_runtime --model_path resize_model --input_dims 2,20,30,40 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model resize_model --iterations 1 --duration 0 --warmup 0 --input_dims 2,20,30,40 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
