#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_batchtospace || (echo "Run sample batchtospace failed"; cd -; exit -1);
command mm_run --magicmind_model batchtospace_model --iterations 1 --duration 0 --warmup 0 --input_dims 4,1,1,3|| (echo "Run mm_run failed"; cd -; exit -1)
command ./sample_runtime --model_path batchtospace_model --input_dims 4,1,1,3|| (echo "Run sample runtime failed"; cd -; exit -1)
cd -