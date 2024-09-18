#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_cast || (echo "Run sample cast failed"; cd -; exit -1);
command ./sample_runtime --model_path cast_model --input_dims 8,4,5,6 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model cast_model --iterations 1 --duration 0 --warmup 0 --input_dims 8,4,5,6 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
