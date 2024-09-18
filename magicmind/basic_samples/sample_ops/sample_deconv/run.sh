#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
command ./sample_deconv || (echo "Run sample deconv failed"; cd -; exit -1);
command ./sample_runtime --model_path deconv_model --input_dims 1,3,2,3 || (echo "Run sample runtime failed"; cd -; exit -1);
command ./mm_run --magicmind_model deconv_model --iterations 1 --duration 0 --warmup 0 --input_dims 1,3,2,3 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
