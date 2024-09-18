#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_reorg || (echo "Run sample reorg failed"; cd -; exit -1);
command ./sample_runtime --model_path reorg_model --input_dims 2,4,4,4 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model reorg_model --iterations 1 --duration 0 --warmup 0 --input_dims 2,4,4,4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
