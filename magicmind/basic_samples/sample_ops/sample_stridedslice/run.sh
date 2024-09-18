#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_stridedslice || (echo "Run sample stridedslice failed"; cd -; exit -1);
command ./sample_runtime --model_path strided_slice_model --input_dims 3,4,5,7 4 4 4 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model strided_slice_model --iterations 1 --duration 0 --warmup 0 --input_dims 3,4,5,7 4 4 4 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
