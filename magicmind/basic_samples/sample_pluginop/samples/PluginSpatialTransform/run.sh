#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
# NCHW broadcast
declare -A map=( ["NCHW"]="2,1,40,180" ["NHWC"]="2,40,180,1")
for layout in "NCHW" "NHWC";
do
  for dtype in "float" "half";
  do
    command ./sample_SpatialTransform --plugin_lib ./libmagicmind_plugin.so --layout $layout --datatype $dtype || (echo "Run sample spatial_transform failed"; cd -; exit -1);
    for broadcast in $(seq 1 2);
    do
      command ./cpu_SpatialTransform --layout $layout --datatype $dtype --input_batches 2,$broadcast,2 --input_dir sample_pluginop/samples/data/plugin_spatial_transformer_op_data/case1/ || (echo "Run cpu spatial_transform failed"; cd -; exit -1);
      command ./sample_runtime --model_path spatial_transform_model_${layout}_${dtype} --input_dims ${map[${layout}]} ${broadcast},6 2,2 --plugin_libs ./libmagicmind_plugin.so --data_path ./input_data ./mat_data ./muta_data || (echo "Run sample runtime failed"; cd -; exit -1);
      command mm_run --magicmind_model spatial_transform_model_${layout}_${dtype} --input_dims ${map[${layout}]} ${broadcast},6 2,2  --iterations 1 --duration 0 --warmup 0 --plugin ./libmagicmind_plugin.so --input_files ./input_data ./mat_data ./muta_data || (echo "Run spatial_transform failed"; cd -; exit -1);
      command diff_compare --data output0 --baseline baseline --datatype $dtype --threshold1 0.002 --threshold2 0.002|| (echo "Run compare failed"; cd -; exit -1);
      command ./cpu_SpatialTransform --layout $layout --datatype $dtype --input_batches 2,$broadcast,2 --input_dir sample_pluginop/samples/data/plugin_spatial_transformer_op_data/case2/ || (echo "Run cpu spatial_transform failed"; cd -; exit -1);
      command ./sample_runtime --model_path spatial_transform_model_${layout}_${dtype} --input_dims ${map[${layout}]} ${broadcast},6 2,2 --plugin_libs ./libmagicmind_plugin.so --data_path ./input_data ./mat_data ./muta_data || (echo "Run sample runtime failed"; cd -; exit -1);
      command mm_run --magicmind_model spatial_transform_model_${layout}_${dtype} --input_dims ${map[${layout}]} ${broadcast},6 2,2  --iterations 1 --duration 0 --warmup 0 --plugin ./libmagicmind_plugin.so --input_files ./input_data ./mat_data ./muta_data || (echo "Run spatial_transform failed"; cd -; exit -1);
      command diff_compare --data output0 --baseline baseline --datatype $dtype --threshold1 0.002 --threshold2 0.002|| (echo "Run compare failed"; cd -; exit -1);
    done
  done
done
cd -
