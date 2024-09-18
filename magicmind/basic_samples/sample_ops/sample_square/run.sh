#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_square || (echo "Run sample square failed"; cd -; exit -1);
command ./sample_runtime --model_path square_model --input_dims 2,4,6,7 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model square_model --iterations 1 --duration 0 --warmup 0 --input_dims 2,4,6,7 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
