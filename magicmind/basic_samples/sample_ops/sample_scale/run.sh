#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
echo $PWD
command ./sample_scale || (echo "Run sample scale failed"; cd -; exit -1);
command ./sample_runtime --model_path scale_model --input_dims 2,3,2 1,1,2 1,1,2 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model scale_model --iterations 1 --duration 0 --warmup 0 --input_dims 2,3,2 1,1,2 1,1,2 || (echo "Run mm_run failed"; cd -; exit -1);
cd -