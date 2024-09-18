#!/bin/bash -x
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_scatter || (echo "Run sample scatter failed"; cd -; exit -1);
command ./sample_runtime --model_path scatter_model --input_dims 12,13,13,14 1,1,1,4 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model scatter_model --iterations 1 --duration 0 --warmup 0 --input_dims 12,13,13,14 1,1,1,4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
