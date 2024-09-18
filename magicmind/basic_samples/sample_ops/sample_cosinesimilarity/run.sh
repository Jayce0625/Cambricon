#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_cosinesimilarity || (echo "Run sample cosinesimilarity failed"; cd -; exit -1);
command ./sample_runtime --model_path cosinesimilarity_model --input_dims 2,6,6,4 2,6,6,4 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model cosinesimilarity_model --iterations 1 --duration 0 --warmup 0 --input_dims 2,6,6,4 2,6,6,4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
