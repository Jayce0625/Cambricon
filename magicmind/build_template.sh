#!/bin/bash
set -x
set -e

EDGE_FLAG=""
WORKSPACE=$(cd $1; pwd)
BUILD_DIR=${WORKSPACE}/build
let BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)/4
export SCRIPT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
# Workspace
if [[ $# == 0 ]]; then
  echo "WORKSPACE path must be provided!"
  exit -1
fi
# Cross compile part
if [[ $# == 2 ]]; then
  TOOLCHAIN_PATH=${TOOLCHAIN_ROOT}
  if [ -z ${TOOLCHAIN_ROOT} ]; then
    TOOLCHAIN_PATH="/usr/local/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"
  fi
  export TOOLCHAIN_PATH=${TOOLCHAIN_PATH}
  echo "Build edge, make sure TOOLCHAIN_ROOT is specific for cross-compile"
  echo "Now is " ${TOOLCHAIN_PATH}
  EDGE_FLAG="-DCMAKE_TOOLCHAIN_FILE=${SCRIPT}/crosscompile.cmake"
  export LD_LIBRARY_PATH="${NEUWARE_HOME}/edge/lib64":$LD_LIBRARY_PATH
fi
# Create build dir
if [ ! -d "${BUILD_DIR}" ]; then
  mkdir "${BUILD_DIR}"
else
  /bin/rm -r ${BUILD_DIR}/
  mkdir "${BUILD_DIR}"
fi
# Build Project
pushd ${BUILD_DIR}
  cmake $EDGE_FLAG \
        ..
popd
cmake --build ${BUILD_DIR} -- -j${BUILD_JOBS}
