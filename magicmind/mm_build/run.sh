#!/bin/bash
set -e
if [[ $# == 0 ]]; then
  echo "Model path must be provided!."
  exit -1
fi
cd $(dirname ${BASH_SOURCE[0]})
run[0]="./caffe_build --prototxt ${1}/ResNet50_train_val_merge_bn.prototxt --caffemodel ${1}/ResNet50_train_val_merge_bn.caffemodel "
run[1]="./onnx_build --onnx ${1}/resnet50-v1-7.onnx "
run[2]="./tensorflow_build --tf_pb ${1}/py3.6.12_resnet50_v1.pb --input_names input --output_names resnet_v1_50/predictions/Reshape_1 "
run[3]="./pytorch_build --pytorch_pt ${1}/py3.7.9_torch1.6.0_resnet50.pt --pt_input_dtypes FLOAT "

run_main() {
  #basic commands
  command ${run[${1}]} || (echo "Run failed"${run[${1}]}; cd -; exit -1);
  #with pre-process
  if [[ ${1} == 2 ]]; then
    command ${run[${1}]} --vars 1,1,1 --means 1,1,1 || (echo "Run failed"${run[${1}]}; cd -; exit -1);
  else
    command ${run[${1}]} --vars 1,1,1 --means 1,1,1 --input_layout NHWC|| (echo "Run failed"${run[${1}]}; cd -; exit -1);
  fi
  #with immutable input shape
  if [[ ${1} == 2 ]]; then
   command ${run[${1}]} --dynamic_shape false --input_dims 4,224,224,3 || (echo "Run failed"${run[${1}]}; cd -; exit -1);
  else
   command ${run[${1}]} --dynamic_shape false --input_dims 4,3,224,224 || (echo "Run failed"${run[${1}]}; cd -; exit -1);
  fi
  #with rgb2bgr
  command ${run[${1}]} --rgb2bgr true || (echo "Run failed"${run[${1}]}; cd -; exit -1);
  #with specific dtype
  command ${run[${1}]} --precision force_float16 --mlu_arch mtp_372,tp_322,mtp_592 --input_dtypes HALF --output_dtypes HALF|| (echo "Run failed"${run[${1}]}; cd -; exit -1);
  #fake calibration
  if [[ ${1} < 2 ]]; then
    command ${run[${1}]} --precision qint8_mixed_float16 --calibration true || (echo "Run failed"${run[${1}]}; cd -; exit -1);
    #calibration with true data
    if [ -L "file_list" ] || [ -d "file_list" ]; then
      command ${run[${1}]} --precision qint8_mixed_float16 --calibration true --file_list ./file_list --calibration_data_path . || (echo "Run failed"${run[${1}]}; cd -; exit -1);
    fi
  elif [[ ${1} == 2 ]]; then
    command ${run[${1}]} --precision qint8_mixed_float16 --input_dims 1,224,224,3 --calibration true || (echo "Run failed"${run[${1}]}; cd -; exit -1);
  else
    command ${run[${1}]} --precision qint8_mixed_float16 --input_dims 1,3,224,224 --calibration true || (echo "Run failed"${run[${1}]}; cd -; exit -1);
    #calibration with true data
    if [ -L "file_list" ] || [ -d "file_list" ]; then
      command ${run[${1}]} --precision qint8_mixed_float16 --input_dims 1,3,224,224 --calibration true --file_list ./file_list --calibration_data_path . || (echo "Run failed"${run[${1}]}; cd -; exit -1);
    fi
  fi
  #bind cluster
  command ${run[${1}]} --mlu_arch mtp_372,tp_322 --cluster_num 2 2 2 2 || (echo "Run failed"${run[${1}]}; cd -; exit -1);
}

for i in $(seq 0 3); do
  run_main ${i} &
  pids[${i}]=$!
done
for pid in ${pids[*]}; do
  wait $pid
done
cd -
