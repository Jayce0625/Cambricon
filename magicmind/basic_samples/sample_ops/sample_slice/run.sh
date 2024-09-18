#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_slice || (echo "Run sample slice failed"; cd -; exit -1);
command ./sample_runtime --model_path slice_model --input_dims 8,4,5,6 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model slice_model --iterations 1 --duration 0 --warmup 0 --input_dims 8,4,5,6 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
