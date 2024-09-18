#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_gru || (echo "Run sample gru failed"; cd -; exit -1);
command ./sample_runtime --model_path gru_model --input_dims 3,2,52 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model gru_model --iterations 1 --duration 0 --warmup 0 --input_dims 3,2,52 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
