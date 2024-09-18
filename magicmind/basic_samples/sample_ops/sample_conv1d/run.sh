#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_conv1d || (echo "Run sample conv1d failed"; cd -; exit -1);
command ./sample_runtime --model_path conv1d_model --input_dims 1,2,3 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model conv1d_model --iterations 1 --duration 0 --warmup 0 --input_dims 1,2,3 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
