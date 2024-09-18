#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_randomuniform || (echo "Run sample randomuniform failed"; cd -; exit -1);
command mm_run --magicmind_model randomuniform_model --iterations 1 --duration 0 --warmup 0 || (echo "Run mm_run failed"; cd -; exit -1);
cd -
