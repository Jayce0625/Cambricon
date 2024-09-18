#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_deformconv2d || (echo "Run sample deformconv2d failed"; cd -; exit -1);
command ./sample_runtime --model_path deformconv2d_model --input_dims 1,2,3,4 1,24,1,1 1,12,1,1 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model deformconv2d_model --iterations 1 --duration 0 --warmup 0 --input_dims 1,2,3,4 1,24,1,1 1,12,1,1 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
