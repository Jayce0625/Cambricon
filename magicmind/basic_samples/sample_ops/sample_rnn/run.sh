#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_rnn || (echo "Run sample rnn failed"; cd -; exit -1);
command ./sample_runtime --model_path rnn_model --input_dims 4,4,374 4,4 128,374 128,128 128,128 128 128 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model rnn_model --iterations 1 --duration 0 --warmup 0 --input_dims 4,4,374 4,4 128,374 128,128 128,128 128 128 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
