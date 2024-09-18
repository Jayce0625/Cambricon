#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_while || (echo "Run sample while failed"; cd -; exit -1);
command mm_run --magicmind_model model_while --iterations 1 --duration 0 --warmup 0 --input_dims 1,3,3,4 1,3,3,4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
