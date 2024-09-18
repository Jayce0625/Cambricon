#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_sin || (echo "Run sample sin failed"; cd -; exit -1);
command ./sample_runtime --model_path sin_model --input_dims 32,64,64,64 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model sin_model --iterations 1 --duration 0 --warmup 0 --input_dims 32,64,64,64 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
