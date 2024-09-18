#!/bin/bash
set -e
if [[ $# == 0 ]]; then
  echo "Model path must be provided!"
  exit -1
fi
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
command ./sample_calibration --data_path . --label_path sample_labels.txt --prototxt_path ${1}/ResNet50_train_val_merge_bn.prototxt --caffemodel_path ${1}/ResNet50_train_val_merge_bn.caffemodel || (echo "Run sample calibration failed"; cd -; exit -1);
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model model_quant --iterations 1 --duration 0 --warmup 0 --input_dims 1,3,224,224 || (echo "Run mm_run failed"; cd -; exit -1);
#Remote
command ./remote_server &
command ./sample_calibration --data_path . --label_path sample_labels.txt --prototxt_path ${1}/ResNet50_train_val_merge_bn.prototxt --caffemodel_path ${1}/ResNet50_train_val_merge_bn.caffemodel --rpc_server localhost:9009 || (echo "Run sample calibration failed"; cd -; exit -1);
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 || (echo "Run sample runtime failed"; cd -; exit -1);
command mm_run --magicmind_model model_quant --iterations 1 --duration 0 --warmup 0 --input_dims 1,3,224,224 || (echo "Run mm_run failed"; cd -; exit -1);
kill -9 $(pgrep -f "./remote_server")
cd -
