#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_matmul || (echo "Run sample matmul failed"; cd -; exit -1);
command ./sample_runtime --model_path matmul_model --input_dims 20,20,12,5 20,20,5,7 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model matmul_model --iterations 1 --duration 0 --warmup 0 --input_dims 20,20,12,5 20,20,5,7 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
