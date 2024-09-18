#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
command ./sample_deconv1d || (echo "Run sample deconv1d failed"; cd -; exit -1);
command ./sample_runtime --model_path deconv1d_model --input_dims 20,6,4 || (echo "Run sample runtime failed"; cd -; exit -1);
command ./mm_run --magicmind_model deconv1d_model --iterations 1 --duration 0 --warmup 0 --input_dims 20,6,4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
