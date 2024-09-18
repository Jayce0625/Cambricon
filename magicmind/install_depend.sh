#!/bin/bash
set -x
set -e
cur_dir=$(pwd)
if [[ "$1" == "install_3226" ]]; then
   cd /tmp/3226
   cp -r /usr/local/neuware/edge ./edge_bak
   rm -rf /usr/local/neuware/edge/*
   chmod +x cntoolkit_edge.run
   ./cntoolkit_edge.run  --nochown
   chmod +x cnnl_edge.run
   ./cnnl_edge.run --target /usr/local/neuware/edge --nochown
   dpkg -X cnnl_static_edge.deb cnnl_static_edge
   cp cnnl_static_edge/usr/local/neuware/lib64/* /usr/local/neuware/edge/lib64/
   chmod +x cnnlextra_edge.run
   ./cnnlextra_edge.run --target /usr/local/neuware/edge --nochown
   chmod +x cnlight_edge.run
   ./cnlight_edge.run --target /usr/local/neuware/edge --nochown
   cp ./edge_bak/bin/mm_run /usr/local/neuware/edge/bin/
   cp ./edge_bak/include/mm_* /usr/local/neuware/edge/include/
   cp -d ./edge_bak/lib64/libmagicmind* /usr/local/neuware/edge/lib64/
   rm -rf ./edge_bak
   export TOOLCHAIN_ROOT="/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"
fi
if [[ "$1" == "install_5223" ]]; then
   cd /tmp/5223
   cp -r /usr/local/neuware/edge ./edge_bak
   rm -rf /usr/local/neuware/edge/*
   chmod +x cntoolkit_edge.run
   ./cntoolkit_edge.run  --nochown
   chmod +x cnnl_edge.run
   ./cnnl_edge.run --target /usr/local/neuware/edge --nochown
   dpkg -X cnnl_static_edge.deb cnnl_static_edge
   cp cnnl_static_edge/usr/local/neuware/lib64/* /usr/local/neuware/edge/lib64/
   chmod +x cnnlextra_edge.run
   ./cnnlextra_edge.run --target /usr/local/neuware/edge --nochown
   chmod +x cnlight_edge.run
   ./cnlight_edge.run --target /usr/local/neuware/edge --nochown
   cp ./edge_bak/bin/mm_run /usr/local/neuware/edge/bin/
   cp ./edge_bak/include/mm_* /usr/local/neuware/edge/include/
   cp -d ./edge_bak/lib64/libmagicmind* /usr/local/neuware/edge/lib64/
   rm -rf ./edge_bak

   if [ -d "/tmp/aarch64--glibc--stable-2020.08-1" ]; then
       rm -rf /tmp/aarch64--glibc--stable-2020.08-1
   fi
   cd /tmp
   tar -xf /tmp/aarch64--glibc--stable-2020.08-1.tar.bz2
   cd /tmp/aarch64--glibc--stable-2020.08-1
   export TOOLCHAIN_ROOT="/tmp/aarch64--glibc--stable-2020.08-1"
fi
cd $cur_dir
