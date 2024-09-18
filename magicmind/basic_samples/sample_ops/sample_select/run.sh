#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_select || (echo "Run sample select failed"; cd -; exit -1);
command ./sample_runtime --model_path Select_model --input_dims 1,2,3,4 1,2,3,4 1|| (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model Select_model --iterations 1 --duration 0 --warmup 0 --input_dims 1,2,3,4 1,2,3,4 1|| (echo "Run mm_run failed"; cd -; exit -1);
cd -
