#! /bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
command ./sample_refit || (echo "Run sample refit failed"; cd -; exit -1);
cd -
