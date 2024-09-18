#!/bin/bash
set -x
set -e
cd $(dirname ${BASH_SOURCE[0]})
SAMPLE_HOME=$PWD
export BASIC_SAMPLE_HOME=${SAMPLE_HOME}/basic_samples/
BUILD_ONLY=false
BUILD_EDGE=false

usage () {
    echo "USAGE: build.sh <options>"
    echo
    echo "OPTIONS:"
    echo "      -e, --edge                                    Build for edge."
    echo "      -b, --build-only"
    echo "      -h, --help                                    Print usage"
}

build_all() {
  if [[ ${BUILD_EDGE} == false ]]; then
    #tools
    if [[ ${IS_KYLIN} != true ]]; then
      pip3 install --upgrade pip
      pip3 install -r ${SAMPLE_HOME}/tools/preprocess/requirements.txt
    fi
    source ${SAMPLE_HOME}/build_template.sh ${SAMPLE_HOME}/tools/diff
    if [[ ! "${PATH[@]}" =~ "${SAMPLE_HOME}/tools/diff/build" ]]; then
      export PATH=${SAMPLE_HOME}/tools/diff/build:$PATH
    fi
    source ${BASIC_SAMPLE_HOME}/build.sh
    #mm_build
    source ${SAMPLE_HOME}/build_template.sh ${SAMPLE_HOME}/mm_build
    source ${SAMPLE_HOME}/build_template.sh ${SAMPLE_HOME}/mm_run
    if [[ ! "${PATH[@]}" =~ "${SAMPLE_HOME}/mm_run/build" ]]; then
      export PATH=${SAMPLE_HOME}/mm_run/build:$PATH
    fi
  else
    source ${BASIC_SAMPLE_HOME}/build.sh --cpu_arch aarch64
    source ${SAMPLE_HOME}/build_template.sh ${SAMPLE_HOME}/mm_run edge
    if [[ ! "${PATH[@]}" =~ "${SAMPLE_HOME}/mm_run/build" ]]; then
      export PATH=${SAMPLE_HOME}/mm_run/build:$PATH
    fi
  fi
}

run_all() {
  if [ -L "data" ] || [ -d "data" ]; then
    IMAGE_DATA=${SAMPLE_HOME}/data
  fi
  if [ -L "model" ] || [ -d "model" ]; then
    MODEL_PATH=${SAMPLE_HOME}/model
  fi
  #basic_samples
  if [ ${MODEL_PATH} ]; then
    python3 ${SAMPLE_HOME}/tools/preprocess/preprocess.py --framework caffe --image_path ${IMAGE_DATA} --save_path ${SAMPLE_HOME}/processed_data --labels ${BASIC_SAMPLE_HOME}/sample_calibration/sample_labels.txt -n 10 -m resnet50
  fi
  #test basic samples
  cd ${BASIC_SAMPLE_HOME}/build
  test_cases=`ls run_*.sh`
  for test_case in $test_cases
  do
    if [[ $test_case != *runtime* ]] && [[ $test_case != *calibration* ]];then
      source $test_case
    fi
  done
  ln -s ${SAMPLE_HOME}/processed_data/* .
  if [ ${MODEL_PATH} ]; then
    cp ${SAMPLE_HOME}/mm_build/build/caffe_build ${SAMPLE_HOME}/basic_samples/build/caffe_build
    source run_sample_runtime.sh ${MODEL_PATH}
    source run_sample_calibration.sh ${MODEL_PATH}
  fi
  #test mm build
  cd ${SAMPLE_HOME}/mm_build/build
  if [ ${MODEL_PATH} ]; then
    ln -s ${SAMPLE_HOME}/processed_data/* .
    source ./run.sh ${MODEL_PATH}
  fi
  cd $SAMPLE_HOME
}

if [ $# != 0 ]; then
  while [ $# != 0 ]; do
    case "$1" in
      -e | --edge)
        shift
        BUILD_EDGE=true
        BUILD_ONLY=true
        ;;
      -b | --build-only)
        shift
        BUILD_ONLY=true
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        echo "-- Unknown options ${1}, use -h or --help"
        usage
        exit -1
        ;;
    esac
  done
fi

build_all
if [[ ${BUILD_ONLY} == false ]]; then
  run_all
fi

unset BASIC_SAMPLE_HOME
