#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_split || (echo "Run sample split failed"; cd -; exit -1);
command ./sample_runtime --model_path split_model --input_dims 2,6,10,10 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model split_model --iterations 1 --duration 0 --warmup 0 --input_dims 2,6,10,10 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
