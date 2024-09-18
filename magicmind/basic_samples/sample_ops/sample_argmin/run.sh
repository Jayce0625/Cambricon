#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_argmin || (echo "Run sample argmin failed"; cd -; exit -1);
command ./sample_runtime --model_path argmin_model --input_dims 1,2,3,4 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model argmin_model --iterations 1 --duration 0 --warmup 0 --input_dims 1,2,3,4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
