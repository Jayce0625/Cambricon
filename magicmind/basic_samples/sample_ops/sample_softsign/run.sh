#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_softsign || (echo "Run sample softsign failed"; cd -; exit -1);
command ./sample_runtime --model_path softsign_model --input_dims 1,224,224,3 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model softsign_model --iterations 1 --duration 0 --warmup 0 --input_dims 1,224,224,3 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
