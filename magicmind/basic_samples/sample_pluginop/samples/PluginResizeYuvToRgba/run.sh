#!/bin/bash
set -e
cd $(dirname ${BASH_SOURCE[0]})
export PATH=$PWD:$PATH
uv_shapes=("1,32,32,1" "2,64,64,1")
y_shapes=("1,64,32,1" "2,128,64,1")
total_batch=("1" "3")
d_col=$((64+$RANDOM%200))
d_row=$((64+$RANDOM%300))
for input_format in $(seq 1 2); do
  for output_format in $(seq 1 6); do
    fill_color_shape=3
    if ((output_format>2)); then
      fill_color_shape=4
    fi
    for pad_method in $(seq 0 1 2); do
      for input_num in $(seq 1 2); do
        ys=()
        uvs=()
        idx_end=$((${input_num}-1))
        for num in $(seq 0 ${idx_end}); do
          ys=(${ys[@]} "./src_y${num}")
          uvs=(${uvs[@]} "./src_uv${num}")
        done
        command ./sample_ResizeYuvToRgba --plugin_lib ./libmagicmind_plugin.so --input_num ${input_num} --input_format ${input_format} --d_row ${d_row} --d_col ${d_row} --total_batch ${total_batch[${idx_end}]} --output_format ${output_format} --pad_method ${pad_method} || (echo "Run sample resize_yuv_to_rgba failed"; cd -; exit -1);
        command ./cpu_ResizeYuvToRgba --uv_shapes ${uv_shapes[@]:0:${input_num}} --d_row ${d_row} --d_col ${d_row} --input_format ${input_format} --output_format ${output_format} --pad_method ${pad_method} --rois 0,10,32,22 || (echo "Run cpu resize_yuv_to_rgba failed"; cd -; exit -1);
        # ys/uvs/inrois/outrois/outshapes/fill_color
        command ./sample_runtime --model_path resize_yuv_to_rgba_model --input_dims ${y_shapes[@]:0:${input_num}} ${uv_shapes[@]:0:${input_num}} ${total_batch[${idx_end}]},4 ${fill_color_shape} --plugin_libs ./libmagicmind_plugin.so --data_path ${ys[@]} ${uvs[@]} ./in_roi ./fill_color || (echo "Run sample runtime failed"; cd -; exit -1);
        command mm_run --magicmind_model resize_yuv_to_rgba_model --input_dims ${y_shapes[@]:0:${input_num}} ${uv_shapes[@]:0:${input_num}} ${total_batch[${idx_end}]},4,1,1 ${fill_color_shape} --iterations 1 --duration 0 --warmup 0 --plugin ./libmagicmind_plugin.so --input_files ${ys[@]} ${uvs[@]} ./in_roi ./fill_color || (echo "Run crop_and_resize failed"; cd -; exit -1);
        command diff_compare --data output0 --baseline baseline --datatype uint8 --threshold1 0.005 --threshold2 0.005 --threshold3 1,1 --threshold4 0.15 || (echo "Run compare failed"; cd -; exit -1);
      done
    done
  done
done
cd -
