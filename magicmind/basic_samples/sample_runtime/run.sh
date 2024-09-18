#!/bin/bash
set -e
if [[ $# == 0 ]]; then
  echo "Model path must be provided!"
  exit -1
fi
cd $(dirname ${BASH_SOURCE[0]})
command ./sample_calibration --data_path . --label_path sample_labels.txt --prototxt_path ${1}/ResNet50_train_val_merge_bn.prototxt --caffemodel_path ${1}/ResNet50_train_val_merge_bn.caffemodel || (echo "Run sample calibration failed"; cd -; exit -1);
command ./caffe_build --prototxt ${1}/ResNet50_train_val_merge_bn.prototxt --caffemodel ${1}/ResNet50_train_val_merge_bn.caffemodel --cluster_num 2 2 2 2 --magicmind_model ./model2 || (echo "Run build failed"${run[${1}]}; cd -; exit -1);
#####
#Local
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 --dev_ids 0 --profile true --dump true --mem_stg static || (echo "Run sample runtime failed"; cd -; exit -1);
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 --dev_ids 0 --profile true --dump true --mem_stg dynamic || (echo "Run sample runtime failed"; cd -; exit -1);
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 --dev_ids 0 --profile true --dump true --mem_stg static --output_caps 4000 || (echo "Run sample runtime failed"; cd -; exit -1);
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 --dev_ids 0 --profile true --dump true --mem_stg dynamic --output_caps 5000 || (echo "Run sample runtime failed"; cd -; exit -1);
#one model with multi-context for binding cluster
command ./sample_runtime --model_path model2 --input_dims 1,3,224,224 --dev_ids 0 --threads 2 --visible_cluster 0,1 2,3 --mem_stg static || (echo "Run sample runtime failed"; cd -; exit -1);
#multi-model for binding cluster
command ./sample_runtime --model_path model2 --input_dims 1,3,224,224 --dev_ids 0 --threads 1 --visible_cluster 0,1 --mem_stg static &
pids[0]=$!
command ./sample_runtime --model_path model2 --input_dims 1,3,224,224 --dev_ids 0 --threads 1 --visible_cluster 2,3 --mem_stg static &
pids[1]=$!
for pid in ${pids[*]}; do
  wait $pid
  return_code=$?
  if [ $return_code -ne 0 ]; then
    (echo "Run build failed"${run[${1}]}; cd -; exit -1);
  fi
done
#Remote
command ./remote_server &
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 --dev_ids 0 --profile true --dump true --mem_stg static --rpc_server localhost:9009 || (echo "Run sample runtime failed"; cd -; kill -9 $(pgrep -f "./remote_server"); exit -1);
command ./sample_runtime --model_path model_quant --input_dims 1,3,224,224 --dev_ids 0 --profile true --dump true --mem_stg dynamic --rpc_server localhost:9009 || (echo "Run sample runtime failed"; cd -; kill -9 $(pgrep -f "./remote_server"); exit -1);
kill -9 $(pgrep -f "./remote_server")
cd -
