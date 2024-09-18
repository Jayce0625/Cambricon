#!/bin/bash
set -x
set -e
set -o pipefail
export WORKSPACE=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
export PATH="/usr/local/ccache:$PATH"

###############################################################################
# Environment Variables
###############################################################################
# BUILD_MODE: release/debug
# BUILD_DIR: build(default)
# BUILD_JOBS: Number of CPU used for project building
# TARGET_MLU_ARCH: CNFATBIN/MLU370/CE3226
# TARGET_CPU_ARCH: x86_64-linux-gnu/i386-linux-gnu/aarch64-linux-gnu/arm-aarch64-gnu/arm-linux-gnueabihf
# TARGET_C_COMPILER: C compiler full-path
# TARGET_CXX_COMPILER: CXX compiler full-path
# TOOLCHAIN_ROOT: /path/to/your/cross-compile-toolchain
# MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP: The list of pluginop names

BUILD_MODE=${BUILD_MODE:-release}
BUILD_DIR=${WORKSPACE}/build
let BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)/4
TARGET_MLU_ARCH=${TARGET_MLU_ARCH:-CNFATBIN}
TARGET_CPU_ARCH=${TARGET_CPU_ARCH:-$(uname -m)-linux-gnu}
TOOLCHAIN_PATH=${TOOLCHAIN_ROOT}
EDGE_FLAG=""
MAGICMIND_SAMPLE="calibration;cfg;quantization;network_modify;refit;runtime;plugin;ops;plugin_crop;"
MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP="SpatialTransform;CropAndResize;ResizeYuvToRgba;"
MAGICMIND_PLUGIN_CROP_SAMPLE="plugin_crop;"
MAGICMIND_OPS_SAMPLE="abs;add;mul;sub;div;argmax;clone;conv1d;conv2d;conv3d;crop;layernorm;logicaland;logicaleq;logicalge;logicalgt;logicalle;logicallt;logicalne;logicalnot;logicalor;logicalxor;reduceall;reduceany;reducemax;reducemean;reduceprod;reducesum;reducenorm1;reducenorm2;reducesumsq;fusedbatchnorm;topk;reverse;reshape;argmin;select;spacetobatch;batchtospace;shufflechannel;log;matrixbandpart;sign;threshold;reorg;groupnorm;selu;softsign;erf;floormod;activation;logsoftmax;lrn;prelu;cos;sin;minimum;deformconv2d;unpool2d;gatherelw;avgpool2d;maxpool2d;pow;squareddifference;sqrt;std;rsqrt;pack;floordiv;exp;elu;asstrided;maximum;normalize;scale;listdiff;pad;split;stridedslice;rnn;transpose;tile;tril;where;clip;floor;avgpool3d;maxpool3d;isfinite;maskedfill;concat;invertpermutation;maskedselect;while;cast;softmax;slice;softplus;randomuniform;cosinesimilarity;square;scatter;resize;matmul;interp;innerproduct;if"
###############################################################################
# Common Funcs
###############################################################################
usage() {
  echo "Usage: ./build.sh <options>"
  echo
  echo "       If need specify neuware path, please:"
  echo "         First, export NEUWARE_HOME=/path/to/where/neuware/installed"
  echo "         Second, export TOOLCHAIN_ROOT=/path/to/cross_toolchain if cross-compile for aarch64-linux-gnu"
  echo
  echo "Options:"
  echo "      -h, --help                       Print usage."
  echo "      -f, --filter[add/calibration/cfg/quantization/network_modify/refit/runtime/plugin/all]"
  echo "                                       Build with specific samples only. Sperated with ';'. Default: all"
  echo "      --cpu_arch[x86_64/aarch64]"
  echo "                                       Build for specific target CPU arch. Default: x86_64."
  echo "                                       Only certain samples are supported for arm."
  echo "      --mlu_arch[322/372]"
  echo "                                       Build for specific target MLU arch. Default: fatbin."
  echo "      -d, --debug                      Build with debug symbols."
  echo "      -v, --verbose                    Build with verbose output."
  echo "      --plugin_filter=[spatial_transform/crop_and_resize/resize_yuv_to_rgba]"
  echo "                                       Build with specific operation only. Sperated with ';'."
  echo
}

###############################################################################
# Build
###############################################################################
# Create build dir
if [ ! -d "${BUILD_DIR}" ]; then
  mkdir "${BUILD_DIR}"
else
  /bin/rm -r ${BUILD_DIR}/
  mkdir "${BUILD_DIR}"
fi

# Handle build-options
if [ $# != 0 ]; then
  while [ $# != 0 ]; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -d | --debug)
        BUILD_MODE="debug"
        shift
        ;;
      -f | --filter)
        shift
        if [[ "$1" =~ "all" ]]; then
          echo "Build all samples."
        else
          MAGICMIND_SAMPLE=$1
        fi
        if [[ "$MAGICMIND_SAMPLE" =~ "plugin" ]]; then
          echo "Build with plugin."
        else
          MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP=""
        fi
        if [[ "$MAGICMIND_SAMPLE" =~ "ops" ]]; then
          echo "Build with op samples."
        else
          MAGICMIND_OPS_SAMPLE=""
        fi
        if [[ "$MAGICMIND_SAMPLE" =~ "plugin_crop" ]]; then
          echo "Build with plugincrop samples."
        else
          MAGICMIND_PLUGIN_CROP_SAMPLE=""
        fi
        shift
        ;;
      --cpu_arch)
        shift
        TARGET_CPU_ARCH=${1}"-linux-gnu"
        shift
        ;;
      --mlu_arch)
        shift
        TARGET_MLU_ARCH="mtp_"${1}
        shift
        ;;
      -v | --verbose)
        BUILD_VERBOSE="VERBOSE=1"
        shift
        ;;
      --plugin_filter)
        shift
        if [[ -n "${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP}" ]]; then
          MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP=$1
        fi
        shift
        ;;
      *)
        echo "Unknown option $1"
        usage
        exit -1
        ;;
    esac
  done
fi

if [[ -n "${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP}" ]]; then
  if [ -z $(which cncc) ]; then
    # cncc is not in PATH, try to search cncc in ${NEUWARE_HOME}/bin.
    export PATH="${NEUWARE_HOME}/bin":$PATH
    if [ -z $(which cncc) ]; then
      echo "-- CNCC cannot be found."
      exit -1
    fi
    cncc --version || ( echo "-- CNCC cannot be used for current CPU target" && exit -1)
  else
    # cncc is already in PATH, but cncc in ${NEUWARE_HOME}/bin is prefered.
    if [ -f "${NEUWARE_HOME}/bin/cncc" ]; then
      ${NEUWARE_HOME}/bin/cncc --version || cannot_use_neuware_cncc=1
      if [ "x${cannot_use_neuware_cncc}" = "x1" ]; then
        echo "-- cncc in ${NEUWARE_HOME}/bin cannot be used for compiling, use default cncc in PATH."
        echo "-- rename cncc libraries in NEUWARE_HOME."
        mv -vf ${NEUWARE_HOME}/lib/clang{,_${TARGET_CPU_ARCH}} || echo "mv clang lib failed."
        mv -vf ${NEUWARE_HOME}/bin{,_${TARGET_CPU_ARCH}} || echo "mv bin failed."
      else
        export PATH="${NEUWARE_HOME}/bin":$PATH
      fi
    fi
  fi
  echo "-- cncc: $(which cncc)"
fi

export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64":$LD_LIBRARY_PATH

# 5. Check compiler and target
[[ ! ${TARGET_CPU_ARCH} =~ $(uname -m) ]] && is_cross_compile=true
IFS='-' read -a target_cpu <<< ${TARGET_CPU_ARCH}
# NOTE variable TARGET_C_COMPILER and TARGET_CXX_COMPILER have higher priority than TOOLCHAIN_ROOT
if [ ! -z ${is_cross_compile} ]; then
  # cross-compiling but toolchain not set, use default path, i.e., MagicMind-devel environment
  case "$(uname -m)-${target_cpu}" in
    x86_64-aarch64)
      if [ -z ${TOOLCHAIN_ROOT} ]; then
        TOOLCHAIN_PATH="/usr/local/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"
      fi
      echo "Only compile libmagicmind_plugin.so and sample_runtime for target aarch64"
      EDGE_FLAG="-DCMAKE_TOOLCHAIN_FILE=${WORKSPACE}/../crosscompile.cmake"
      export LD_LIBRARY_PATH="${NEUWARE_HOME}/edge/lib64":$LD_LIBRARY_PATH
      MAGICMIND_SAMPLE="runtime;"
      MAGICMIND_OPS_SAMPLE=""
      ;;
    *)
      echo "Unsupported target cpu arch."
      exit -1
      ;;
  esac
fi

if [ ! -z ${TOOLCHAIN_PATH} ]; then
  export TOOLCHAIN_PATH=${TOOLCHAIN_PATH}
fi

# Build Project
pushd ${BUILD_DIR}
  cmake $EDGE_FLAG \
        -DCMAKE_BUILD_TYPE="${BUILD_MODE}" \
        -DTARGET_MLU_ARCH="${TARGET_MLU_ARCH}" \
        -DTARGET_CPU_ARCH="${TARGET_CPU_ARCH}" \
        -DMAGICMIND_PLUGIN_BUILD_SPECIFIC_OP="${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP}" \
        -DMAGICMIND_PLUGIN_CROP_SAMPLE="${MAGICMIND_PLUGIN_CROP_SAMPLE}" \
        -DMAGICMIND_SAMPLE="${MAGICMIND_SAMPLE}" \
        -DMAGICMIND_OPS_SAMPLE="${MAGICMIND_OPS_SAMPLE}" \
        ..
popd
cmake --build ${BUILD_DIR} -- ${BUILD_VERBOSE} -j${BUILD_JOBS}
