/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_RESIZE_YUV_TO_RGBA_MLU200_KERNEL_H_
#define PLUGIN_RESIZE_YUV_TO_RGBA_MLU200_KERNEL_H_
#include "plugin_resize_yuv_to_rgba_macro.h"
#if __BANG_ARCH__ >= 200 && __BANG_ARCH__ < 300
__mlu_shared__ uint8_t sram_buffer[SRAM_SIZE];
__mlu_shared__ int sync_buffer[128];
__nram__ int8_t nram_buffer[NRAM_SIZE];

__wram__ int16_t yuv_filter_wram[kNumFilterChOut * 2 * kNumFilterChIn];
__wram__ int8_t copy_filter_wram[kNumMultLimit * kNumLt * kNumLt];

/*------------------------------ HELP FUNCTIONS ------------------------------*/
// load convert filter and bias from gdram
__mlu_func__ void loadCvtFilter(half *yuv_filter_wram,
                                half *yuv_bias_nram,
                                half *yuv_filter_gdram,
                                half *yuv_bias_gdram) {
  // load filter and bias for yuv2rgba conv
  __memcpy(yuv_bias_nram, yuv_bias_gdram, kNumFilterChOut * sizeof(half), GDRAM2NRAM);
  __memcpy(yuv_filter_wram, yuv_filter_gdram, 2 * kNumFilterChIn * kNumFilterChOut * sizeof(half),
           GDRAM2WRAM);
}

// load mult copy filter from gdram if needed
// mult: the multipiler used in upscaling mode
__mlu_func__ void loadCopyFilter(int8_t *copy_filter_wram, int8_t *copy_filter_addr, int mult) {
  // load copy filter if need src expansion
  if (mult > 1 && mult <= kNumMultLimit) {
    __memcpy(copy_filter_wram, copy_filter_addr, kNumLt * kNumLt * mult * sizeof(int8_t),
             GDRAM2WRAM);
  }
}

// find w nram max load num
// mult: the multipiler used in upscaling mode
__mlu_func__ int nram_find_limit(int src_roi_w_,
                                 int src_roi_w,
                                 int dst_roi_w,
                                 int mult,
                                 int dst_channel) {
  int const_size = HALF_PAD_SIZE * sizeof(half) +   // acuracy compensation
                   HALF_PAD_SIZE * sizeof(half) +   // threshold max
                   HALF_PAD_SIZE * sizeof(half) +   // half2uint8 temp
                   kNumFilterChOut * sizeof(half);  // yuv2rgba bias

  int lower_bound = 1;
  int upper_bound = src_roi_w_ / 2 - 1;

  int limit = BANG_MAX(upper_bound, 1);

  if (mult == 1) {
    while (lower_bound < upper_bound - 1) {
      int load_w  = (limit + 1) * 2;
      int store_w = __float2int_up((float)load_w * dst_roi_w / src_roi_w);

      // rgba should align to 2 ct lines
      int load_w_pad = CEIL_ALIGN(load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
      int store_rgbx_pad = CEIL_ALIGN(store_w * dst_channel, HALF_PAD_SIZE);

      // 2 point(left + right)
      int mask_pad    = CEIL_ALIGN((load_w + 1) * kNumDefaultChannel, HALF_PAD_SIZE);
      int mask_size   = 2 * mask_pad * sizeof(half);
      int weight_size = store_rgbx_pad * sizeof(half);

      // 2line y or 2 line uv should align to kNumFilterChIn
      int yuv_2line_pad = CEIL_ALIGN(load_w_pad * 2, kNumFilterChIn);

      // origin yuv input(2 line y and 2 line uv) align to UINT8_PAD_SIZE
      // or 1 line left and right interp rgba pixel
      int ram_size1 = BANG_MAX(CEIL_ALIGN(2 * yuv_2line_pad, UINT8_PAD_SIZE), 2 * store_rgbx_pad) *
                      sizeof(half);

      // rgba input(2 line)
      int ram_size2 = 2 * load_w_pad * kNumDefaultChannel * sizeof(half);

      int malloc_size = mask_size + weight_size + ram_size1 + ram_size2;

      if (malloc_size <= NRAM_SIZE - const_size)
        lower_bound = limit;
      else
        upper_bound = limit;

      limit = (lower_bound + upper_bound) / 2;
    }
  } else {  // mult != 1
    while (lower_bound < upper_bound - 1) {
      int load_w  = (limit + 1) * 2;
      int store_w = __float2int_up((float)load_w * dst_roi_w / src_roi_w);

      // rgba should align to 2 ct lines
      int load_w_pad = CEIL_ALIGN(load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
      int store_rgbx_pad = CEIL_ALIGN(store_w * dst_channel, HALF_PAD_SIZE);

      // 2 point(left + right)
      int mask_pad    = CEIL_ALIGN(mult * (load_w + 1) * kNumDefaultChannel, HALF_PAD_SIZE);
      int mask_size   = 2 * mask_pad * sizeof(half);
      int weight_size = store_rgbx_pad * sizeof(half);

      // 2line y or 2 line uv should align to kNumFilterChIn
      int yuv_2line_pad = CEIL_ALIGN(load_w_pad * 2, kNumFilterChIn);

      // 1 line rgba input after mult
      int input_mult_rgba = BANG_MAX(mult * load_w_pad * kNumDefaultChannel, mask_pad);

      // origin yuv input(2 line y and 2 line uv) align to UINT8_PAD_SIZE
      // or 2 line rgba input after mult
      int ram_size1 =
          BANG_MAX(CEIL_ALIGN(2 * yuv_2line_pad, UINT8_PAD_SIZE), input_mult_rgba) * sizeof(half);

      // rgba input(2 line) or 1 line and left and right interp pixel
      int ram_size2 = 2 * BANG_MAX(load_w_pad * kNumDefaultChannel, store_rgbx_pad) * sizeof(half);

      int malloc_size = mask_size + weight_size + ram_size1 + ram_size2;

      if (malloc_size <= NRAM_SIZE - const_size)
        lower_bound = limit;
      else
        upper_bound = limit;

      limit = (lower_bound + upper_bound) / 2;
    }
  }  // mult != 1

  return limit * 2;
}

// find h sram max load num
__mlu_func__ int sram_find_limit(int src_roi_h,
                                 int dst_roi_h,
                                 int max_load_w,
                                 int max_store_w,
                                 int dst_channel) {
  int lower_bound = 1;
  int upper_bound = CEIL_ALIGN(dst_roi_h, coreDim) / coreDim;

  int limit = upper_bound;

  // float src_dst_scale_h = float(src_roi_h) / dst_roi_h;
  int src_dst_scale_h_int    = src_roi_h / dst_roi_h;
  float src_dst_scale_h_frac = float(src_roi_h - src_dst_scale_h_int * dst_roi_h) / dst_roi_h;

  while (lower_bound < upper_bound - 1) {
    int max_store_h_sram = limit * coreDim;
    int max_load_h_sram  = __float2int_up((max_store_h_sram - 1) * src_dst_scale_h_int +
                                         (max_store_h_sram - 1) * src_dst_scale_h_frac) +
                          2;

    int max_input_y_sram  = max_load_h_sram * max_load_w;
    int max_input_uv_sram = (max_load_h_sram / 2 + 1) * max_load_w;

    int max_output_sram = max_store_h_sram * max_store_w * dst_channel;

    int sram_malloc_size =
        (max_input_y_sram + max_input_uv_sram + max_output_sram) * sizeof(uint8_t) * 2;

    if (sram_malloc_size <= SRAM_SIZE)
      lower_bound = limit;
    else
      upper_bound = limit;

    limit = (lower_bound + upper_bound) / 2;
  }

  return (limit * coreDim);
}

// 1.move data from SRAM to NRAM
// 2.compute rgbx output
// 3.move output from NRAM to SRAM
__mlu_func__ void mvAndComputeStage(uint8_t *input_y_sram,
                                    uint8_t *input_uv_sram,
                                    uint8_t *output_sram,
                                    int16_t *yuv_filter_wram,
                                    half *yuv_bias_nram,
                                    int8_t *copy_filter_wram,
                                    half *round_nram,
                                    half *max_value_nram,
                                    half *cvt_temp_nram,
                                    half *mask_left_nram,
                                    half *mask_right_nram,
                                    half *weight_nram,
                                    half *ram1_nram,
                                    half *ram2_nram,
                                    uint8_t *load_y_nram,
                                    int load_w,
                                    int load_mask,
                                    int mult,
                                    int store_h_sram,
                                    int dst_idx_h_start,
                                    int src_dst_scale_h_int,
                                    float src_dst_scale_h_frac,
                                    float src_pos_h_offset,
                                    int src_roi_h,
                                    int src_roi_y,
                                    int src_y_h_sram_start,
                                    int src_uv_h_sram_start,
                                    int store_rgbx) {
  // compute handle size
  int load_w_pad = CEIL_ALIGN(load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
  int rgba_pad   = load_w_pad * kNumDefaultChannel;

  // 2 line y or 2 line uv should align to kNumFilterChIn(conv inst require)
  int yuv_2line_pad = CEIL_ALIGN(load_w_pad * 2, kNumFilterChIn);

  // compute instruction need to pad 2 line
  int load_mask_pad   = CEIL_ALIGN(load_mask + mult * kNumDefaultChannel, HALF_PAD_SIZE);
  int load_weight     = store_rgbx;
  int load_weight_pad = CEIL_ALIGN(load_weight, HALF_PAD_SIZE);

  // nram remap, put load y and load uv in second half
  half *y_nram          = (half *)ram1_nram;
  uint8_t *load_uv_nram = (uint8_t *)load_y_nram + yuv_2line_pad;

  // Multi-core related params
  int store_h_nram_seg    = store_h_sram / coreDim;
  int store_h_nram_rem    = store_h_sram % coreDim;
  int store_h_nram_deal   = store_h_nram_seg + (coreId < store_h_nram_rem ? 1 : 0);
  int store_h_nram_offset = store_h_nram_seg * coreId + BANG_MIN(coreId, store_h_nram_rem);

  for (int h_iter = 0; h_iter < store_h_nram_deal; ++h_iter) {
    /*############################################*/
    /*#[Module]: loadTwoLines(Four in YUV mode)#*/
    /*############################################*/
    // compute dst and src h
    int dst_idx_h = dst_idx_h_start + store_h_nram_offset + h_iter;
    float src_pos_h =
        dst_idx_h * src_dst_scale_h_int + dst_idx_h * src_dst_scale_h_frac + src_pos_h_offset;

    src_pos_h         = BANG_MIN(src_pos_h, src_roi_h - 1);
    int src_pos_h_int = __float2int_dn(src_pos_h) * (int)(src_pos_h > 0);

    // compute offsets for each row
    // if src_dst_scale_h < 2
    int y1_h       = src_pos_h_int + src_roi_y;
    int y2_h       = BANG_MIN(src_pos_h_int + 1, src_roi_h - 1) + src_roi_y;
    int uv1_h      = y1_h / 2;
    int uv2_h      = y2_h / 2;
    int y1_offset  = (y1_h - src_y_h_sram_start) * load_w;
    int y2_offset  = (y2_h - src_y_h_sram_start) * load_w;
    int uv1_offset = (uv1_h - src_uv_h_sram_start) * load_w;
    int uv2_offset = (uv2_h - src_uv_h_sram_start) * load_w;

    if (src_dst_scale_h_int >= 2) {
      y1_h       = (store_h_nram_offset + h_iter) * 2;
      y2_h       = y1_h + 1;
      uv1_h      = (store_h_nram_offset + h_iter) * 2;
      uv2_h      = uv1_h + 1;
      y1_offset  = y1_h * load_w;
      y2_offset  = y2_h * load_w;
      uv1_offset = uv1_h * load_w;
      uv2_offset = uv2_h * load_w;
    }

    // Load two lines of Y and two line of UV
    __memcpy((uint8_t *)load_y_nram, (uint8_t *)input_y_sram + y1_offset, load_w * sizeof(uint8_t),
             SRAM2NRAM);

    __memcpy((uint8_t *)load_uv_nram, (uint8_t *)input_uv_sram + uv1_offset,
             load_w * sizeof(uint8_t), SRAM2NRAM);

    __memcpy((uint8_t *)load_y_nram + load_w_pad, (uint8_t *)input_y_sram + y2_offset,
             load_w * sizeof(uint8_t), SRAM2NRAM);

    __memcpy((uint8_t *)load_uv_nram + load_w_pad, (uint8_t *)input_uv_sram + uv2_offset,
             load_w * sizeof(uint8_t), SRAM2NRAM);

    /*#################################*/
    /*#[Module]: Preprocess(For YUV)#*/
    /*#################################*/
    // convert uint8 yuv data to half
    __bang_uchar2half((half *)y_nram, (unsigned char *)load_y_nram,
                      CEIL_ALIGN(2 * yuv_2line_pad, UINT8_PAD_SIZE));

    // convert half yuv data to int16(position is -7)
    __bang_half2int16_rd((int16_t *)y_nram, (half *)y_nram, 2 * yuv_2line_pad, -7);

    /*######################*/
    /*#[Module]: YUV2RGB0#*/
    /*######################*/
    /* Fold input YUV data so that the input channel becomes kNumFilterChIn
     * Memory is continuous in this (-->) direction (in_width /= kNumFilterChIn)
     * [Y1,1 Y1,2 ... Y1,31 Y1,32] ... [Y2,1 Y2,2 ... Y2,31 Y2,32] ...
     * [U1,1 V1,1 ... U1,16 V1,16] ... [U2,1 U2,2 ... U2,16 V2,16] ...
     * Input shape: 1(N) x 2(H) x yuv_2line_pad/kNumFilterChIn(W) x kNumFilterChIn(C)
     *
     * For each kNumFilterChIn of input, we need 4 X kNumFilterChIn kernels to convert
     * kNumFilterChIn gray pixcels into 4 X kNumFilterChIn RGBA pixcels. Each kernel has
     * a shape of: 1 x 2 x 1 x kNumFilterChIn. For example,
     * [ 1.164  0     0 ... 0] -> 1.164 * Y1,1 -> R1,1
     * [ 0      1.586   ... 0]  + 1.586 * V1,1
     * [ 1.164  0     0 ... 0] -> 1.164 * Y1,1 -> G1,1
     * [-0.392 -0.813 0 ... 0]  - 0.392 * U1,1
     *                          - 0.813 * V1,1
     * ...
     * ...
     * [ 0 0 1.164 0     0 ... 0] -> 1.164 * Y1,3 -> R1,3
     * [ 0 0 0     1.586 0 ... 0]  + 1.586 * V1,2
     * ...
     * Total 4 X kNumFilterChIn pixcels hence 4 X kNumFilterChIn kernels
     */
    half *rgba_nram = (half *)ram2_nram;

    // conv params in order
    // in_channel = kNumFilterChIn;
    // in_height = 2;
    // in_width = yuv_2line_pad / kNumFilterChIn;
    // filter_height = 2;
    // filter_width = 1;
    // stride_height = 1;
    // stride_width = 1;
    // out_channel = kNumFilterChOut;
    __bang_conv((half *)rgba_nram, (int16_t *)y_nram, (int16_t *)yuv_filter_wram,
                (half *)yuv_bias_nram, kNumFilterChIn, 2, yuv_2line_pad / kNumFilterChIn, 2, 1, 1,
                1, kNumFilterChOut, -20);

    // truncate values < 0
    __bang_active_relu((half *)rgba_nram, (half *)rgba_nram, 2 * rgba_pad);

    // truncate values > 255
    __bang_cycle_minequal((half *)rgba_nram, (half *)rgba_nram, (half *)max_value_nram,
                          2 * rgba_pad, HALF_PAD_SIZE);

    /*################################################*/
    /*#[Module] Interpolate top line and bottom line #*/
    /*################################################*/
    half h_weight       = (half)(src_pos_h - src_pos_h_int) * (int)(src_pos_h > 0);
    half *top_rgba_nram = (half *)rgba_nram;
    half *bot_rgba_nram = top_rgba_nram + rgba_pad;

    // temp1 = (1-h_weight)*top + h_weight*bot
    __bang_sub(bot_rgba_nram, bot_rgba_nram, top_rgba_nram, rgba_pad);
    __bang_mul_scalar(bot_rgba_nram, bot_rgba_nram, h_weight, rgba_pad);
    __bang_add(top_rgba_nram, top_rgba_nram, bot_rgba_nram, rgba_pad);

    /*#################################*/
    /*#[Module]: src image expansion#*/
    /*#################################*/
    half *src_rgba_nram    = (half *)rgba_nram;
    half *select_rgba_nram = (half *)ram1_nram;

    if (mult > 1 && mult <= kNumMultLimit) {
      src_rgba_nram    = (half *)ram1_nram;
      select_rgba_nram = (half *)ram2_nram;

      __bang_conv((int16_t *)src_rgba_nram, (int16_t *)rgba_nram, (int8_t *)copy_filter_wram,
                  kNumLt, 1, rgba_pad / kNumLt, 1, 1, 1, 1, kNumLt * mult, 0);
    } else if (mult > kNumMultLimit) {
      src_rgba_nram    = (half *)ram1_nram;
      select_rgba_nram = (half *)ram2_nram;

      for (int m = 0; m < mult; m++) {
        __memcpy((half *)src_rgba_nram + m * kNumDefaultChannel, (half *)rgba_nram,
                 kNumDefaultChannel * sizeof(half),         // size
                 NRAM2NRAM,                                 // direction
                 mult * kNumDefaultChannel * sizeof(half),  // dst stride
                 kNumDefaultChannel * sizeof(half),         // src stride
                 load_w_pad - 1);                           // seg number
      }
    }

    // Select data using the mask gegerated in [1]
    // For example,
    /* Before:
     * [Y0X0 Y0X1] ... [Y0X4 Y0X5] ... [Y0X8 Y0X9] ...
     * [Y1X0 Y1X1] ... [Y1X4 Y1X5] ... [Y1X8 Y1X9] ...
     *  .    .          .    .          .    .
     *
     * After:
     * Y0X0 Y0X4 Y0X8 ... Y0X1 Y0X5 Y0X9 ...
     * Y1X0 Y1X4 Y1X8 ... Y1X1 Y1X5 Y1X9 ...
     * .    .    .        .    .    .
     */
    half *select_left_nram  = select_rgba_nram;
    half *select_right_nram = select_left_nram + load_weight_pad;
    __bang_write_zero(select_right_nram, load_weight_pad);

    __bang_collect(select_left_nram, src_rgba_nram, mask_left_nram, load_mask_pad);
    __bang_collect(select_right_nram, src_rgba_nram, mask_right_nram, load_mask_pad);

    /*################################################*/
    /*#[Module] Interpolate left line and right line #*/
    /*################################################*/
    // res =  (1-w_weight)*left+w_weight*right
    __bang_sub(select_right_nram, select_right_nram, select_left_nram, load_weight_pad);
    __bang_mul(select_right_nram, select_right_nram, weight_nram, load_weight_pad);
    __bang_add(select_left_nram, select_left_nram, select_right_nram, load_weight_pad);

    /*###############################################################*/
    /*#[Module 7]: Postprocess && Store Data#*/
    /*###############################################################*/
    half *dst_nram      = select_rgba_nram;
    half *dst_mask_nram = dst_nram + load_weight_pad;

    // acuracy compensation
    // __bang_cycle_add((half *)dst_nram, (half *)dst_nram,
    //                  (half *)round_nram, load_weight_pad, HALF_PAD_SIZE);

    // half to uint8
    __bang_cycle_ge((half *)dst_mask_nram, (half *)dst_nram, (half *)cvt_temp_nram, load_weight_pad,
                    HALF_PAD_SIZE);

    __bang_mul_scalar((half *)dst_mask_nram, (half *)dst_mask_nram, 256, load_weight_pad);

    __bang_sub((half *)dst_mask_nram, (half *)dst_nram, (half *)dst_mask_nram, load_weight_pad);

    // acuracy compensation
    // __bang_cycle_add((half *)dst_mask_nram, (half *)dst_mask_nram,
    //                  (half *)round_nram, load_weight_pad, HALF_PAD_SIZE);

    __bang_add_scalar((half *)dst_mask_nram, (half *)dst_mask_nram, (half)0.5, load_weight_pad);
    __bang_half2uchar_dn((signed char *)dst_nram, (half *)dst_mask_nram,
                         CEIL_ALIGN(load_weight_pad, UINT8_PAD_SIZE));

    __memcpy((uint8_t *)output_sram + (dst_idx_h - dst_idx_h_start) * store_rgbx,
             (uint8_t *)dst_nram,
             store_rgbx * sizeof(uint8_t),       // size
             NRAM2SRAM,                          // direction
             store_rgbx * sizeof(uint8_t),       // dst stride
             load_weight_pad * sizeof(uint8_t),  // src stride
             0);                                 // seg number
  }                                              // h_iter
}

// pipeline mode
__mlu_func__ void resizeConvertPipelineMode(uint32_t max_load_w,
                                            uint32_t max_store_w,
                                            uint32_t dst_channel,
                                            uint32_t src_roi_x,
                                            uint32_t src_roi_x_,
                                            uint32_t src_roi_w,
                                            uint32_t src_roi_y,
                                            uint32_t src_stride,
                                            uint32_t dst_stride,
                                            int src_roi_h,
                                            int dst_roi_h,
                                            int dst_roi_w,
                                            int max_mask_num,
                                            int max_weight_num,
                                            int max_2line_yuv_pad,
                                            int mult,
                                            int load_w_repeat,
                                            int load_w_remain,
                                            int src_dst_scale_h_int,
                                            float src_dst_scale_h_frac,
                                            float dst_src_scale_w,
                                            float dst_pos_w_offset,
                                            float src_pos_h_offset,
                                            uint8_t *sram_buffer,
                                            int *sync_buffer,
                                            uint8_t *src_y_addr,
                                            uint8_t *src_uv_addr,
                                            uint8_t *dst_addr,
                                            int16_t *yuv_filter_wram,
                                            half *yuv_bias_nram,
                                            int8_t *copy_filter_wram,
                                            half *mask_nram,
                                            half *weight_nram,
                                            half *ram1_nram,
                                            half *ram2_nram,
                                            half *mask_left_nram,
                                            half *mask_right_nram,
                                            half *mask_left_addr,
                                            half *mask_right_addr,
                                            half *weight_right_addr,
                                            half *round_nram,
                                            half *max_value_nram,
                                            half *cvt_temp_nram) {
  int src_idx_w       = 0 - (src_roi_x % 2);
  int dst_idx_w_start = 0;
  int dst_idx_w_end   = 0;

  int dst_idx_h_start = 0;
  int dst_idx_h_end   = -1;
  int src_idx_h_start = 0;
  int src_idx_h_end   = 0;

  int src_y_h_sram_start  = 0;
  int src_y_h_sram_end    = 0;
  int src_uv_h_sram_start = 0;
  int src_uv_h_sram_end   = 0;
  int load_y_h_sram       = 0;
  int load_uv_h_sram      = 0;

  /*############################################*/
  /*#[Module]: SRAM REMAP and h partition      #*/
  /*############################################*/
  // Sram h partition
  // if src_dst_scale_h >= 2
  if (coreId == 0x00) {
    sync_buffer[0] = max_load_w;
    sync_buffer[1] = max_store_w;
  }
  __sync_cluster();
  max_load_w  = sync_buffer[0];
  max_store_w = sync_buffer[1];
  __sync_cluster();

  int max_store_h_sram = coreDim;
  int max_load_h_sram  = max_store_h_sram * 2;

  // max input y and uv sm size
  int max_input_y_sram  = max_load_h_sram * max_load_w;
  int max_input_uv_sram = max_load_h_sram * max_load_w;

  int max_output_sram = max_store_h_sram * max_store_w * dst_channel;

  if (src_dst_scale_h_int < 2) {
    max_store_h_sram = sram_find_limit(src_roi_h, dst_roi_h, max_load_w, max_store_w, dst_channel);
    max_load_h_sram  = __float2int_up((max_store_h_sram - 1) * src_dst_scale_h_int +
                                     (max_store_h_sram - 1) * src_dst_scale_h_frac) +
                      2;

    if (coreId == 0x00) {
      sync_buffer[0] = max_store_h_sram;
      sync_buffer[1] = max_load_h_sram;
    }
    __sync_cluster();
    max_store_h_sram = sync_buffer[0];
    max_load_h_sram  = sync_buffer[1];
    __sync_cluster();

    max_input_y_sram  = max_load_h_sram * max_load_w;
    max_input_uv_sram = (max_load_h_sram / 2 + 1) * max_load_w;

    max_output_sram = max_store_h_sram * max_store_w * dst_channel;
  }

  // Memory SRAM usage
  uint8_t *input_y_ping_sram  = (uint8_t *)sram_buffer;
  uint8_t *input_uv_ping_sram = (uint8_t *)input_y_ping_sram + max_input_y_sram;
  uint8_t *output_ping_sram   = (uint8_t *)input_uv_ping_sram + max_input_uv_sram;

  // divide dst_roi_h to
  // store_h_sram_repeat * max_store_h_sram + store_h_sram_remain
  int store_h_sram_repeat = dst_roi_h / max_store_h_sram;
  int store_h_sram_remain = dst_roi_h % max_store_h_sram;

  // cur roi y and uv addr
  uint8_t *cur_y_addr  = src_y_addr + src_roi_x_;
  uint8_t *cur_uv_addr = src_uv_addr + src_roi_x_;

  for (int w_iter = 0; w_iter < load_w_repeat + 1 /* remainder */; w_iter++) {
    /* src_roi_w part
     * | w_load_num_00 | w_load_num_01| ... | w_load_remaind_0 |
     *   .               .                    .
     *   .               .                    .
     *   .               .                    .
     * | w_load_num_N0 | w_load_num_N1| ... | w_load_remaind_N |
     *
     * Note: w_load_num_00 ... w_load_num_N0 share the same mask and weights
     * so the order of LOAD/COMPUTE/STORE looks like:
     * |   /|   /|  ...
     * |  / |  / |  ...
     * | /  | /  |  ...
     * |/   |/   |/ ...
     */

    /*################################*/
    /*#[Module]: loadMaskAndWeights  #*/
    /*################################*/
    // mask load num each w iter
    int load_w        = (w_iter < load_w_repeat) ? max_load_w : load_w_remain + 2;
    int compute_w     = (w_iter < load_w_repeat) ? max_load_w - 2 : load_w;
    int load_mask     = mult * kNumDefaultChannel * compute_w;
    int load_mask_pad = CEIL_ALIGN(load_mask + mult * kNumDefaultChannel, HALF_PAD_SIZE);

    // load left mask and right mask
    __bang_write_value(mask_left_nram, load_mask_pad, 0);
    __bang_write_value(mask_right_nram, load_mask_pad, 0);

    __memcpy((half *)mask_left_nram, (half *)mask_left_addr, load_mask * sizeof(half), GDRAM2NRAM);
    __memcpy(((half *)mask_right_nram + mult * kNumDefaultChannel), (half *)mask_right_addr,
             load_mask * sizeof(half), GDRAM2NRAM);

    // current mask addr fix
    mask_left_addr += load_mask;
    mask_right_addr += load_mask;

    // update src w idx
    src_idx_w = src_idx_w + compute_w;

    // compute dst w idx end
    dst_idx_w_end = __float2int_up(((float)src_idx_w + 0.5) * dst_src_scale_w - 0.5);
    dst_idx_w_end = BANG_MIN(dst_idx_w_end, dst_roi_w);

    // compute dst store w
    int store_rgbx = (dst_idx_w_end - dst_idx_w_start) * dst_channel;

    // update dst w idx start
    dst_idx_w_start = dst_idx_w_end;

    if (store_rgbx > 0) {
      // weight load num each w iter
      int load_weight = store_rgbx;

      // load left weight and right weight
      __memcpy((half *)weight_nram, (half *)weight_right_addr, load_weight * sizeof(half),
               GDRAM2NRAM);

      // current weight addr fix
      weight_right_addr += load_weight;

      // NRAM REMAP
      // nram remap, put load y and load uv in second half
      uint8_t *load_y_nram = (uint8_t *)((half *)ram1_nram + max_2line_yuv_pad);

/*###############################*/
/*#[Module ]: H Partition       #*/
/*###############################*/
#define SRAM_LOAD_INFO_UPDATE dst_idx_h_start = dst_idx_h_end + 1;

#define SRAM_LOAD_INFO_COMPUTE(store_h_sram)                                                     \
  dst_idx_h_end = dst_idx_h_start + store_h_sram - 1;                                            \
  if (src_dst_scale_h_int < 2) {                                                                 \
    src_idx_h_start = __float2int_dn(dst_idx_h_start * src_dst_scale_h_int +                     \
                                     dst_idx_h_start * src_dst_scale_h_frac + src_pos_h_offset); \
    src_idx_h_start = BANG_MAX(src_idx_h_start, 0);                                              \
    src_idx_h_start = BANG_MIN(src_idx_h_start, src_roi_h - 1);                                  \
    src_idx_h_end   = __float2int_dn(dst_idx_h_end * src_dst_scale_h_int +                       \
                                   dst_idx_h_end * src_dst_scale_h_frac + src_pos_h_offset) +  \
                    1;                                                                           \
    src_idx_h_end       = BANG_MIN(src_idx_h_end, src_roi_h - 1);                                \
    src_y_h_sram_start  = src_idx_h_start + src_roi_y;                                           \
    src_y_h_sram_end    = src_idx_h_end + src_roi_y;                                             \
    src_uv_h_sram_start = src_y_h_sram_start / 2;                                                \
    src_uv_h_sram_end   = src_y_h_sram_end / 2;                                                  \
    load_y_h_sram       = src_idx_h_end - src_idx_h_start + 1;                                   \
    load_uv_h_sram      = src_uv_h_sram_end - src_uv_h_sram_start + 1;                           \
  }

#define SRAM_LOAD(input_y_sram, input_uv_sram, store_h_sram)                                       \
  if (src_dst_scale_h_int < 2) {                                                                   \
    if (coreId == 0x00) {                                                                          \
      sync_buffer[0] = src_y_h_sram_start;                                                         \
      sync_buffer[1] = src_uv_h_sram_start;                                                        \
      sync_buffer[2] = load_y_h_sram;                                                              \
      sync_buffer[3] = load_uv_h_sram;                                                             \
    }                                                                                              \
    __sync_cluster();                                                                              \
    src_y_h_sram_start  = sync_buffer[0];                                                          \
    src_uv_h_sram_start = sync_buffer[1];                                                          \
    load_y_h_sram       = sync_buffer[2];                                                          \
    load_uv_h_sram      = sync_buffer[3];                                                          \
    __sync_cluster();                                                                              \
    __memcpy_async(input_y_sram, (uint8_t *)cur_y_addr + src_y_h_sram_start * src_stride,          \
                   load_w * sizeof(uint8_t), GDRAM2SRAM, load_w * sizeof(uint8_t), src_stride,     \
                   load_y_h_sram - 1);                                                             \
    __memcpy_async(input_uv_sram, (uint8_t *)cur_uv_addr + src_uv_h_sram_start * src_stride,       \
                   load_w * sizeof(uint8_t), GDRAM2SRAM, load_w * sizeof(uint8_t), src_stride,     \
                   load_uv_h_sram - 1);                                                            \
  } else {                                                                                         \
    for (int store_h_sram_iter = 0; store_h_sram_iter < store_h_sram; store_h_sram_iter++) {       \
      int dst_idx_h_sram = dst_idx_h_start + store_h_sram_iter;                                    \
      int src_idx_h_sram =                                                                         \
          __float2int_dn(dst_idx_h_sram * src_dst_scale_h_int +                                    \
                         dst_idx_h_sram * src_dst_scale_h_frac + src_pos_h_offset);                \
      src_idx_h_sram     = BANG_MIN(src_idx_h_sram, src_roi_h - 1);                                \
      int src_y_h0_sram  = src_idx_h_sram + src_roi_y;                                             \
      int src_uv_h0_sram = src_y_h0_sram / 2;                                                      \
      src_idx_h_sram     = BANG_MIN(src_idx_h_sram + 1, src_roi_h - 1);                            \
      int src_y_h1_sram  = src_idx_h_sram + src_roi_y;                                             \
      int src_uv_h1_sram = src_y_h1_sram / 2;                                                      \
      if (coreId == 0x00) {                                                                        \
        sync_buffer[0] = src_y_h0_sram;                                                            \
        sync_buffer[1] = src_uv_h0_sram;                                                           \
        sync_buffer[2] = src_y_h1_sram;                                                            \
        sync_buffer[3] = src_uv_h1_sram;                                                           \
      }                                                                                            \
      __sync_cluster();                                                                            \
      src_y_h0_sram  = sync_buffer[0];                                                             \
      src_uv_h0_sram = sync_buffer[1];                                                             \
      src_y_h1_sram  = sync_buffer[2];                                                             \
      src_uv_h1_sram = sync_buffer[3];                                                             \
      __sync_cluster();                                                                            \
      /*if (coreId != 0x80) {*/                                                                    \
      __memcpy_async(input_y_sram + store_h_sram_iter * 2 * load_w,                                \
                     (uint8_t *)cur_y_addr + src_y_h0_sram * src_stride, load_w * sizeof(uint8_t), \
                     GDRAM2SRAM);                                                                  \
      __memcpy_async(input_uv_sram + store_h_sram_iter * 2 * load_w,                               \
                     (uint8_t *)cur_uv_addr + src_uv_h0_sram * src_stride,                         \
                     load_w * sizeof(uint8_t), GDRAM2SRAM);                                        \
      __memcpy_async(input_y_sram + (store_h_sram_iter * 2 + 1) * load_w,                          \
                     (uint8_t *)cur_y_addr + src_y_h1_sram * src_stride, load_w * sizeof(uint8_t), \
                     GDRAM2SRAM);                                                                  \
      __memcpy_async(input_uv_sram + (store_h_sram_iter * 2 + 1) * load_w,                         \
                     (uint8_t *)cur_uv_addr + src_uv_h1_sram * src_stride,                         \
                     load_w * sizeof(uint8_t), GDRAM2SRAM);                                        \
      /*} */                                                                                       \
    }                                                                                              \
  }

#define MV_COMPUTE(input_y_sram, input_uv_sram, output_sram, store_h_sram)                      \
  mvAndComputeStage((input_y_sram), (input_uv_sram), (output_sram), (int16_t *)yuv_filter_wram, \
                    (half *)yuv_bias_nram, (int8_t *)copy_filter_wram, (half *)round_nram,      \
                    (half *)max_value_nram, (half *)cvt_temp_nram, (half *)mask_left_nram,      \
                    (half *)mask_right_nram, (half *)weight_nram, (half *)ram1_nram,            \
                    (half *)ram2_nram, (uint8_t *)load_y_nram, load_w, load_mask, mult,         \
                    store_h_sram, dst_idx_h_start, src_dst_scale_h_int, src_dst_scale_h_frac,   \
                    src_pos_h_offset, src_roi_h, src_roi_y, src_y_h_sram_start,                 \
                    src_uv_h_sram_start, store_rgbx);

#define SRAM_STORE(output_gdram, output_sram, store_h_sram)                             \
  if (coreId == 0x80) {                                                                 \
    __memcpy_async(output_gdram, output_sram, store_rgbx * sizeof(uint8_t), SRAM2GDRAM, \
                   dst_stride, store_rgbx * sizeof(uint8_t), store_h_sram - 1);         \
  }

      dst_idx_h_end = -1;

      if (store_h_sram_repeat > 0) {
        // LOAD
        SRAM_LOAD_INFO_UPDATE;

        SRAM_LOAD_INFO_COMPUTE(max_store_h_sram);

        SRAM_LOAD(((uint8_t *)input_y_ping_sram), ((uint8_t *)input_uv_ping_sram),
                  max_store_h_sram);

        __sync_cluster();
      }

      if (store_h_sram_repeat > 1) {
        // COMPUTE
        MV_COMPUTE((uint8_t *)input_y_ping_sram, (uint8_t *)input_uv_ping_sram,
                   (uint8_t *)output_ping_sram, max_store_h_sram);

        // LOAD
        SRAM_LOAD_INFO_UPDATE;

        SRAM_LOAD_INFO_COMPUTE(max_store_h_sram);

        SRAM_LOAD(((uint8_t *)input_y_ping_sram + SRAM_PONG),
                  ((uint8_t *)input_uv_ping_sram + SRAM_PONG), max_store_h_sram);

        __sync_cluster();
      }

      for (int i = 0; i < store_h_sram_repeat - 2; ++i) {
        // COMPUTE
        MV_COMPUTE((uint8_t *)input_y_ping_sram + ((i + 1) % 2) * SRAM_PONG,
                   (uint8_t *)input_uv_ping_sram + ((i + 1) % 2) * SRAM_PONG,
                   (uint8_t *)output_ping_sram + ((i + 1) % 2) * SRAM_PONG, max_store_h_sram);

        // STORE
        SRAM_STORE((uint8_t *)dst_addr + i * max_store_h_sram * dst_stride,
                   (uint8_t *)output_ping_sram + (i % 2) * SRAM_PONG, max_store_h_sram);

        // LOAD
        SRAM_LOAD_INFO_UPDATE;

        SRAM_LOAD_INFO_COMPUTE(max_store_h_sram);

        SRAM_LOAD(((uint8_t *)input_y_ping_sram + (i % 2) * SRAM_PONG),
                  ((uint8_t *)input_uv_ping_sram + (i % 2) * SRAM_PONG), max_store_h_sram);

        __sync_cluster();
      }

      if (store_h_sram_repeat > 0) {
        // COMPUTE
        MV_COMPUTE((uint8_t *)input_y_ping_sram + (store_h_sram_repeat + 1) % 2 * SRAM_PONG,
                   (uint8_t *)input_uv_ping_sram + (store_h_sram_repeat + 1) % 2 * SRAM_PONG,
                   (uint8_t *)output_ping_sram + (store_h_sram_repeat + 1) % 2 * SRAM_PONG,
                   max_store_h_sram);
      }

      if (store_h_sram_repeat > 1) {
        // STORE
        SRAM_STORE((uint8_t *)dst_addr + (store_h_sram_repeat - 2) * max_store_h_sram * dst_stride,
                   (uint8_t *)output_ping_sram + (store_h_sram_repeat % 2) * SRAM_PONG,
                   max_store_h_sram);
      }

      if (store_h_sram_remain > 0) {
        // LOAD
        SRAM_LOAD_INFO_UPDATE;

        SRAM_LOAD_INFO_COMPUTE(store_h_sram_remain);

        SRAM_LOAD(((uint8_t *)input_y_ping_sram + (store_h_sram_repeat % 2) * SRAM_PONG),
                  ((uint8_t *)input_uv_ping_sram + (store_h_sram_repeat % 2) * SRAM_PONG),
                  store_h_sram_remain);
      }

      __sync_cluster();

      if (store_h_sram_remain > 0) {
        // COMPUTE
        MV_COMPUTE((uint8_t *)input_y_ping_sram + (store_h_sram_repeat % 2) * SRAM_PONG,
                   (uint8_t *)input_uv_ping_sram + (store_h_sram_repeat % 2) * SRAM_PONG,
                   (uint8_t *)output_ping_sram + (store_h_sram_repeat % 2) * SRAM_PONG,
                   store_h_sram_remain);
      }

      if (store_h_sram_repeat > 0) {
        // STORE
        SRAM_STORE((uint8_t *)dst_addr + (store_h_sram_repeat - 1) * max_store_h_sram * dst_stride,
                   (uint8_t *)output_ping_sram + ((store_h_sram_repeat - 1) % 2) * SRAM_PONG,
                   max_store_h_sram);
      }

      if (store_h_sram_remain > 0) {
        __sync_cluster();

        // STORE
        SRAM_STORE((uint8_t *)dst_addr + store_h_sram_repeat * max_store_h_sram * dst_stride,
                   (uint8_t *)output_ping_sram + (store_h_sram_repeat % 2) * SRAM_PONG,
                   store_h_sram_remain);
      }

      __sync_cluster();
    }  // store_rgbx > 0

    cur_y_addr += compute_w;
    cur_uv_addr += compute_w;
    dst_addr += store_rgbx;
  }  // for w_iter
}

// no pipeline mode
__mlu_func__ void resizeConvertNoPipelineMode(uint32_t max_load_w,
                                              uint32_t max_store_w,
                                              uint32_t dst_channel,
                                              uint32_t src_roi_x,
                                              uint32_t src_roi_x_,
                                              uint32_t src_roi_w,
                                              uint32_t src_roi_y,
                                              uint32_t src_stride,
                                              uint32_t dst_stride,
                                              int src_roi_h,
                                              int dst_roi_h,
                                              int dst_roi_w,
                                              int max_mask_num,
                                              int max_weight_num,
                                              int max_2line_yuv_pad,
                                              int mult,
                                              int load_w_repeat,
                                              int load_w_remain,
                                              int src_dst_scale_h_int,
                                              float src_dst_scale_h_frac,
                                              float dst_src_scale_w,
                                              float dst_pos_w_offset,
                                              float src_pos_h_offset,
                                              uint8_t *src_y_addr,
                                              uint8_t *src_uv_addr,
                                              uint8_t *dst_addr,
                                              int16_t *yuv_filter_wram,
                                              half *yuv_bias_nram,
                                              int8_t *copy_filter_wram,
                                              half *mask_nram,
                                              half *weight_nram,
                                              half *ram1_nram,
                                              half *ram2_nram,
                                              half *mask_left_nram,
                                              half *mask_right_nram,
                                              half *mask_left_addr,
                                              half *mask_right_addr,
                                              half *weight_right_addr,
                                              half *round_nram,
                                              half *max_value_nram,
                                              half *cvt_temp_nram) {
  if (coreId == 0x80) {
    return;
  }

  int src_idx_w       = 0 - (src_roi_x % 2);
  int dst_idx_w_start = 0;
  int dst_idx_w_end   = 0;

  int dst_idx_h_start = 0;

  // Multi-core related params
  int dst_roi_h_seg    = dst_roi_h / coreDim;
  int dst_roi_h_rem    = dst_roi_h % coreDim;
  int dst_roi_h_deal   = dst_roi_h_seg + (coreId < dst_roi_h_rem ? 1 : 0);
  int dst_roi_h_offset = dst_roi_h_seg * coreId + (coreId < dst_roi_h_rem ? coreId : dst_roi_h_rem);

  int seg_offset = dst_roi_h_offset * dst_stride;
  dst_addr       = dst_addr + seg_offset;

  uint8_t *cur_y_addr  = src_y_addr;
  uint8_t *cur_uv_addr = src_uv_addr;

  for (int w_iter = 0; w_iter < load_w_repeat + 1 /* remainder */; w_iter++) {
    uint8_t *dst_store_addr = dst_addr;
    /* src_roi_w part
     * | w_load_num_00 | w_load_num_01| ... | w_load_remaind_0 |
     *   .               .                    .
     *   .               .                    .
     *   .               .                    .
     * | w_load_num_N0 | w_load_num_N1| ... | w_load_remaind_N |
     *
     * Note: w_load_num_00 ... w_load_num_N0 share the same mask and weights
     * so the order of LOAD/COMPUTE/STORE looks like:
     * |   /|   /|  ...
     * |  / |  / |  ...
     * | /  | /  |  ...
     * |/   |/   |/ ...
     */

    /*################################*/
    /*#[Module]: loadMaskAndWeights  #*/
    /*################################*/
    // mask load num each w iter
    int load_w        = (w_iter < load_w_repeat) ? max_load_w : load_w_remain + 2;
    int compute_w     = (w_iter < load_w_repeat) ? max_load_w - 2 : load_w;
    int load_mask     = mult * kNumDefaultChannel * compute_w;
    int load_mask_pad = CEIL_ALIGN(load_mask + mult * kNumDefaultChannel, HALF_PAD_SIZE);

    // load left mask and right mask
    __bang_write_value(mask_left_nram, load_mask_pad, 0);
    __bang_write_value(mask_right_nram, load_mask_pad, 0);

    __memcpy((half *)mask_left_nram, (half *)mask_left_addr, load_mask * sizeof(half), GDRAM2NRAM);
    __memcpy(((half *)mask_right_nram + mult * kNumDefaultChannel), (half *)mask_right_addr,
             load_mask * sizeof(half), GDRAM2NRAM);

    // current mask addr fix
    mask_left_addr += load_mask;
    mask_right_addr += load_mask;

    // update src w idx
    src_idx_w = src_idx_w + compute_w;

    // compute dst w idx end
    dst_idx_w_end = __float2int_up(((float)src_idx_w + 0.5) * dst_src_scale_w - 0.5);
    dst_idx_w_end = BANG_MIN(dst_idx_w_end, dst_roi_w);

    // compute dst store w
    int store_rgbx = (dst_idx_w_end - dst_idx_w_start) * dst_channel;

    // update dst w idx start
    dst_idx_w_start = dst_idx_w_end;

    if (store_rgbx > 0) {
      // weight load num each w iter
      int load_weight     = store_rgbx;
      int load_weight_pad = CEIL_ALIGN(load_weight, HALF_PAD_SIZE);

      // load left weight and right weight
      __memcpy((half *)weight_nram, (half *)weight_right_addr, load_weight * sizeof(half),
               GDRAM2NRAM);

      // current weight addr fix
      weight_right_addr += load_weight;

      // current h weight
      half h_weight = 0.0;
      for (int h_iter = 0; h_iter < dst_roi_h_deal; ++h_iter) {
        int dst_idx_h = dst_idx_h_start + dst_roi_h_offset + h_iter;
        float src_pos_h =
            dst_idx_h * src_dst_scale_h_int + dst_idx_h * src_dst_scale_h_frac + src_pos_h_offset;

        src_pos_h         = BANG_MIN(src_pos_h, src_roi_h - 1);
        int src_pos_h_int = __float2int_dn(src_pos_h) * (int)(src_pos_h > 0);

        // compute offsets for each row
        int y1_h       = src_pos_h_int + src_roi_y;
        int y2_h       = BANG_MIN(src_pos_h_int + 1, src_roi_h - 1) + src_roi_y;
        int uv1_h      = y1_h / 2;
        int uv2_h      = y2_h / 2;
        int y1_offset  = y1_h * src_stride + src_roi_x_;
        int y2_offset  = y2_h * src_stride + src_roi_x_;
        int uv1_offset = uv1_h * src_stride + src_roi_x_;
        int uv2_offset = uv2_h * src_stride + src_roi_x_;

        /*#################################*/
        /*#[Module]: Preprocess(For YUV)#*/
        /*#################################*/
        int load_w_pad =
            CEIL_ALIGN(load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
        int yuv_2line_pad = CEIL_ALIGN(load_w_pad * 2, kNumFilterChIn);
        int rgba_pad      = load_w_pad * kNumDefaultChannel;

        half *y_nram       = (half *)ram1_nram;
        char *load_y_nram  = (char *)((half *)ram1_nram + max_2line_yuv_pad);
        char *load_uv_nram = (char *)load_y_nram + yuv_2line_pad;

        __memcpy((char *)load_y_nram, (char *)cur_y_addr + y1_offset, load_w * sizeof(char),
                 GDRAM2NRAM);
        __memcpy((char *)load_uv_nram, (char *)cur_uv_addr + uv1_offset, load_w * sizeof(char),
                 GDRAM2NRAM);

        __memcpy((char *)load_y_nram + load_w_pad, (char *)cur_y_addr + y2_offset,
                 load_w * sizeof(char), GDRAM2NRAM);
        __memcpy((char *)load_uv_nram + load_w_pad, (char *)cur_uv_addr + uv2_offset,
                 load_w * sizeof(char), GDRAM2NRAM);

        // convert uint8 yuv data to half
        __bang_uchar2half((half *)y_nram, (unsigned char *)load_y_nram,
                          CEIL_ALIGN(2 * yuv_2line_pad, UINT8_PAD_SIZE));

        // convert half yuv data to int16(position is -7)
        __bang_half2int16_rd((int16_t *)y_nram, (half *)y_nram, 2 * yuv_2line_pad, -7);

        half *rgba_nram = (half *)ram2_nram;

        // convert yuv data to rgba data
        __bang_conv((half *)rgba_nram, (int16_t *)y_nram, (int16_t *)yuv_filter_wram,
                    (half *)yuv_bias_nram, kNumFilterChIn, 2, yuv_2line_pad / kNumFilterChIn, 2, 1,
                    1, 1, kNumFilterChOut, -20);

        // truncate values < 0
        __bang_active_relu(rgba_nram, rgba_nram, 2 * rgba_pad);

        // truncate values > 255
        __bang_cycle_minequal(rgba_nram, rgba_nram, max_value_nram, 2 * rgba_pad, HALF_PAD_SIZE);

        /*################################################*/
        /*#[Module] Interpolate top line and bottom line #*/
        /*################################################*/
        h_weight            = (half)(src_pos_h - src_pos_h_int) * (int)(src_pos_h > 0);
        half *top_rgba_nram = (half *)rgba_nram;
        half *bot_rgba_nram = top_rgba_nram + rgba_pad;

        // temp1 = (1-h_weight)*top + h_weight*bot
        __bang_sub(bot_rgba_nram, bot_rgba_nram, top_rgba_nram, rgba_pad);
        __bang_mul_scalar(bot_rgba_nram, bot_rgba_nram, h_weight, rgba_pad);
        __bang_add(top_rgba_nram, top_rgba_nram, bot_rgba_nram, rgba_pad);

        half *src_rgba_nram    = (half *)rgba_nram;
        half *select_rgba_nram = (half *)ram1_nram;

        if (mult > 1 && mult <= kNumMultLimit) {
          src_rgba_nram    = (half *)ram1_nram;
          select_rgba_nram = (half *)ram2_nram;

          __bang_conv((int16_t *)src_rgba_nram, (int16_t *)rgba_nram, (int8_t *)copy_filter_wram,
                      kNumLt, 1, rgba_pad / kNumLt, 1, 1, 1, 1, kNumLt * mult, 0);
        } else if (mult > kNumMultLimit) {
          src_rgba_nram    = (half *)ram1_nram;
          select_rgba_nram = (half *)ram2_nram;

          for (int m = 0; m < mult; m++) {
            __memcpy((half *)src_rgba_nram + m * kNumDefaultChannel, (half *)rgba_nram,
                     kNumDefaultChannel * sizeof(half),         // size
                     NRAM2NRAM,                                 // direction
                     mult * kNumDefaultChannel * sizeof(half),  // dst stride
                     kNumDefaultChannel * sizeof(half),         // src stride
                     load_w_pad - 1);                           // seg number
          }
        }

        // Select data using the mask gegerated
        // For example,
        /* Before:
         * [Y0X0 Y0X1] ... [Y0X4 Y0X5] ... [Y0X8 Y0X9] ...
         * [Y1X0 Y1X1] ... [Y1X4 Y1X5] ... [Y1X8 Y1X9] ...
         *  .    .          .    .          .    .
         *  .    .          .    .          .    .
         *
         * After:
         * Y0X0 Y0X4 Y0X8 ... Y0X1 Y0X5 Y0X9 ...
         * Y1X0 Y1X4 Y1X8 ... Y1X1 Y1X5 Y1X9 ...
         * .    .    .        .    .    .
         * .    .    .        .    .    .
         */
        half *select_left_nram  = (half *)select_rgba_nram;
        half *select_right_nram = select_left_nram + load_weight_pad;

        __bang_write_zero(select_right_nram, load_weight_pad);

        __bang_collect(select_left_nram, src_rgba_nram, mask_left_nram, load_mask_pad);
        __bang_collect(select_right_nram, src_rgba_nram, mask_right_nram, load_mask_pad);

        /*#################################################*/
        /*#[Module]. Interpolate left line and right line #*/
        /*#################################################*/
        // res =  (1-w_weight)*left+w_weight*right
        __bang_sub(select_right_nram, select_right_nram, select_left_nram, load_weight_pad);
        __bang_mul(select_right_nram, select_right_nram, weight_nram, load_weight_pad);
        __bang_add(select_left_nram, select_left_nram, select_right_nram, load_weight_pad);

        /*##################################################################*/
        /*#[Module]: Postprocess && Store Data#*/
        /*##################################################################*/
        half *dst_nram      = select_rgba_nram;
        half *dst_mask_nram = dst_nram + load_weight_pad;

        __bang_cycle_ge((half *)dst_mask_nram, (half *)dst_nram, (half *)cvt_temp_nram,
                        load_weight_pad, HALF_PAD_SIZE);
        __bang_mul_scalar((half *)dst_mask_nram, (half *)dst_mask_nram, 256, load_weight_pad);
        __bang_sub((half *)dst_mask_nram, (half *)dst_nram, (half *)dst_mask_nram, load_weight_pad);
        __bang_add_scalar((half *)dst_mask_nram, (half *)dst_mask_nram, (half)0.5, load_weight_pad);
        __bang_half2uchar_dn((signed char *)dst_nram, (half *)dst_mask_nram,
                             CEIL_ALIGN(load_weight_pad, UINT8_PAD_SIZE));

        __memcpy(dst_store_addr, (unsigned char *)dst_nram, store_rgbx * sizeof(uint8_t),
                 NRAM2GDRAM, store_rgbx * sizeof(uint8_t), load_weight_pad * sizeof(uint8_t), 0);
        dst_store_addr += dst_stride;
      }  // for h_iter
    }    // store_rgbx > 0
    cur_y_addr += compute_w;
    cur_uv_addr += compute_w;
    dst_addr += store_rgbx;
  }  // for w_iter
}

__mlu_func__ void strategyOfHeightPartitionCore(const int32_t height,
                                                const bool is_batch_split,
                                                int32_t &h_core_num,
                                                int32_t &h_core_start) {
  if (is_batch_split) {
    int32_t height_seg = height / coreDim;
    int32_t height_rem = height % coreDim;
    h_core_num         = height_seg + (coreId < height_rem ? 1 : 0);
    h_core_start       = coreId * height_seg + (coreId < height_rem ? coreId : height_rem);
  } else {
    int32_t height_seg = height / taskDim;
    int32_t height_rem = height % taskDim;
    h_core_num         = height_seg + (taskId < height_rem ? 1 : 0);
    h_core_start       = taskId * height_seg + (taskId < height_rem ? taskId : height_rem);
  }
}

__mlu_func__ void fillBackgroundColor(uint8_t *dst_addr,
                                      uint8_t *pad_nram,
                                      uint8_t *round_nram,
                                      const bool is_batch_split,
                                      const int32_t fill_color_align,
                                      const uint32_t dst_width,
                                      const uint32_t dst_height,
                                      const uint32_t dst_channel,
                                      const uint32_t dst_stride,
                                      const uint32_t dst_roi_x,
                                      const uint32_t dst_roi_y,
                                      const uint32_t dst_roi_w,
                                      const uint32_t dst_roi_h) {
  if (dst_roi_w != dst_width && dst_roi_h == dst_height) {
    if (dst_roi_x == 0) {
      int32_t pad_w          = (dst_width - dst_roi_w) * dst_channel;
      int32_t div_w          = CEIL_ALIGN(pad_w, fill_color_align) / fill_color_align;
      int32_t pad_line_start = 0;
      int32_t pad_line_num   = 0;
      __memcpy(pad_nram, round_nram, fill_color_align, NRAM2NRAM, fill_color_align, 0, div_w - 1);
      strategyOfHeightPartitionCore(dst_height, is_batch_split, pad_line_num, pad_line_start);
      if (pad_line_num > 0) {
        __memcpy(dst_addr + pad_line_start * dst_stride + dst_roi_w * dst_channel, pad_nram, pad_w,
                 NRAM2GDRAM, dst_stride, 0, pad_line_num - 1);
      }
    } else {
      int32_t pad_w_left  = dst_roi_x * dst_channel;
      int32_t pad_w_right = (dst_width - dst_roi_x - dst_roi_w) * dst_channel;
      int32_t div_w =
          CEIL_ALIGN(BANG_MAX(pad_w_left, pad_w_right), fill_color_align) / fill_color_align;
      int32_t pad_line_start = 0;
      int32_t pad_line_num   = 0;
      __memcpy(pad_nram, round_nram, fill_color_align, NRAM2NRAM, fill_color_align, 0, div_w - 1);
      strategyOfHeightPartitionCore(dst_height, is_batch_split, pad_line_num, pad_line_start);
      if (pad_line_num > 0) {
        if (pad_w_left > 0) {
          __memcpy(dst_addr + pad_line_start * dst_stride, pad_nram, pad_w_left, NRAM2GDRAM,
                   dst_stride, 0, pad_line_num - 1);
        }
        if (pad_w_right > 0) {
          __memcpy(dst_addr + pad_line_start * dst_stride + (dst_roi_w + dst_roi_x) * dst_channel,
                   pad_nram, pad_w_right, NRAM2GDRAM, dst_stride, 0, pad_line_num - 1);
        }
      }
    }
  } else if (dst_roi_w == dst_width && dst_roi_h != dst_height) {
    if (dst_roi_y == 0) {
      int32_t pad_w          = dst_width * dst_channel;
      int32_t div_w          = CEIL_ALIGN(pad_w, fill_color_align) / fill_color_align;
      int32_t pad_line_start = 0;
      int32_t pad_line_num   = 0;
      int32_t pad_h          = dst_height - dst_roi_h;
      __memcpy(pad_nram, round_nram, fill_color_align, NRAM2NRAM, fill_color_align, 0, div_w - 1);
      strategyOfHeightPartitionCore(pad_h, is_batch_split, pad_line_num, pad_line_start);
      pad_line_start += dst_roi_h;
      if (pad_line_num > 0) {
        __memcpy(dst_addr + pad_line_start * dst_stride, pad_nram, pad_w, NRAM2GDRAM, dst_stride, 0,
                 pad_line_num - 1);
      }
    } else {
      int32_t pad_w       = 0;
      int32_t pad_w_start = 0;
      int32_t pad_h_top   = dst_roi_y;
      int32_t pad_h_bot   = dst_height - dst_roi_y - dst_roi_h;
      if (is_batch_split) {
        int32_t pad_seg = dst_width / coreDim;
        int32_t pad_rem = dst_width % coreDim;
        pad_w           = pad_seg + (coreId < pad_rem ? 1 : 0);
        pad_w_start     = coreId * pad_seg + (coreId < pad_rem ? coreId : pad_rem);
      } else {
        int32_t pad_seg = dst_width / taskDim;
        int32_t pad_rem = dst_width % taskDim;
        pad_w           = pad_seg + (taskId < pad_rem ? 1 : 0);
        pad_w_start     = taskId * pad_seg + (taskId < pad_rem ? taskId : pad_rem);
      }
      pad_w *= dst_channel;
      pad_w_start *= dst_channel;
      int32_t div_w = CEIL_ALIGN(pad_w, fill_color_align) / fill_color_align;
      if (div_w > 0) {
        __memcpy(pad_nram, round_nram, fill_color_align, NRAM2NRAM, fill_color_align, 0, div_w - 1);
      }
      if (pad_h_top > 0 && div_w > 0) {
        __memcpy(dst_addr + pad_w_start, pad_nram, pad_w, NRAM2GDRAM, dst_stride, 0, pad_h_top - 1);
      }
      if (pad_h_bot > 0 && div_w > 0) {
        __memcpy(dst_addr + (dst_roi_y + dst_roi_h) * dst_stride + pad_w_start, pad_nram, pad_w,
                 NRAM2GDRAM, dst_stride, 0, pad_h_bot - 1);
      }
    }
  }
}
/*---------------------------- MLU ENTRY FUNCTION ----------------------------*/
/*!
 *  @brief A function.
 *
 *  A fusionOp of resize and yuv2rgbx
 *
 *  @param[in]  src_gdram
 *    Input. src image Y and UV channel addrs in gdram.
 *  @param[out] dst_gdram
 *    Output. dst image addrs in gdram.
 *  @param[in]  batch_size
 *    Input. batch size.
 *  @param[in]  convert_filter_gdram
 *    Input. the filter used to do yuv2rgb conversion addrs in gdram.
 *  @param[in]  convert_bias_gdram
 *    Input. the bias needed by yuv2rgb conversion addrs in gdram.
 *  @param[in]  shape_gdram
 *    Input. the shape of source and destination shape infomation addrs in
 * gdram.
 *  @param[in]  mask_gdram
 *    Input. the mask addrs used to select src pixel addrs in gdram.
 *  @param[in]  weight_gdram
 *    Input. the weight used to do bilinear interp addrs in gdram.
 *  @param[in]  copy_filter_gdram
 *    Input. the filter used to using conv to mult src pixels addrs in gdram.
 */
__mlu_global__ void MLUUnion1KernelResizeConvert200(void **src_y_gdram,
                                                    void **src_uv_gdram,
                                                    void **dst_gdram,
                                                    void *fill_color_gdram,
                                                    uint32_t batch_size,
                                                    uint32_t dst_channel,
                                                    void *convert_filter_gdram,
                                                    void *convert_bias_gdram,
                                                    void *shape_gdram,
                                                    void **mask_gdram,
                                                    void **weight_gdram,
                                                    void **copy_filter_gdram) {
  /**---------------------- Initialization ----------------------**/
  // Memory usage
  // Put all const variables(bias, weight, mask) at front of the buffer
  // so that space after them can be used freely without concerning
  // overwritting const variables
  half *round_nram     = (half *)nram_buffer;
  half *max_value_nram = round_nram + HALF_PAD_SIZE;
  half *cvt_temp_nram  = max_value_nram + HALF_PAD_SIZE;
  half *yuv_bias_nram  = cvt_temp_nram + HALF_PAD_SIZE;

  // fill background color
  uint8_t *fill_color_nram  = (uint8_t *)(yuv_bias_nram + kNumFilterChOut);
  int32_t fill_color_align  = 0;
  int32_t fill_color_repeat = 0;
  if (dst_channel == 3) {
    fill_color_repeat = DDR_ALIGN_SIZE;
    fill_color_align  = DDR_ALIGN_SIZE * dst_channel;
  } else if (dst_channel == 4) {
    fill_color_repeat = DDR_ALIGN_SIZE / dst_channel;
    fill_color_align  = DDR_ALIGN_SIZE;
  }
  for (int32_t i = 0; i < fill_color_repeat; ++i) {
    for (int cidx = 0; cidx < dst_channel; cidx++) {
      fill_color_nram[dst_channel * i + cidx] = ((unsigned char *)fill_color_gdram)[cidx];
    }
  }

  uint8_t *pad_nram       = fill_color_nram + fill_color_align;
  half *compute_temp_nram = (half *)pad_nram;

  // init round_nram data
  __bang_write_value(round_nram, HALF_PAD_SIZE, 0.5);

  // init max value data
  __bang_write_value(max_value_nram, HALF_PAD_SIZE, 255);

  // init convert temp data
  __bang_write_value(cvt_temp_nram, HALF_PAD_SIZE, 128);

  // load yuv2rgba filter and bias
  loadCvtFilter((half *)yuv_filter_wram, (half *)yuv_bias_nram, (half *)convert_filter_gdram,
                (half *)convert_bias_gdram);

  // batch segment
  int batch_seg = batch_size / taskDimY;
  int batch_rem = batch_size % taskDimY;

  int start_batch = taskIdY * batch_seg + (taskIdY < batch_rem ? taskIdY : batch_rem);
  int end_batch   = start_batch + batch_seg + (taskIdY < batch_rem ? 1 : 0);

  // for batch_size
  for (int batch = start_batch; batch < end_batch; batch++) {
    // each batch param mlu addr
    uint8_t *src_y_addr  = (uint8_t *)(src_y_gdram[batch]);
    uint8_t *src_uv_addr = (uint8_t *)(src_uv_gdram[batch]);
    uint8_t *dst_addr    = (uint8_t *)(dst_gdram[batch]);

    half *mask_left_addr     = (half *)(mask_gdram[batch * 2]);
    half *mask_right_addr    = (half *)(mask_gdram[batch * 2 + 1]);
    half *weight_right_addr  = (half *)(weight_gdram[batch]);
    int8_t *copy_filter_addr = (int8_t *)(copy_filter_gdram[batch]);

    // each batch param
    uint32_t src_stride = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 0];
    uint32_t src_roi_x  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 2];
    uint32_t src_roi_y  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 3];
    uint32_t src_roi_w  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 4];
    uint32_t src_roi_h  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 5];

    uint32_t src_roi_x_ = src_roi_x / 2 * 2;
    uint32_t src_roi_w_ = (src_roi_x % 2 + src_roi_w + 1) / 2 * 2;

    uint32_t dst_stride = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 6];
    uint32_t dst_height = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 7];
    uint32_t dst_roi_x  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 8];
    uint32_t dst_roi_y  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 9];
    uint32_t dst_roi_w  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 10];
    uint32_t dst_roi_h  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 11];

    if (coreId != 0x80) {
      fillBackgroundColor(dst_addr, pad_nram, fill_color_nram, 1, fill_color_align,
                          dst_stride / dst_channel, dst_height, dst_channel, dst_stride, dst_roi_x,
                          dst_roi_y, dst_roi_w, dst_roi_h);
    }

    // dst_addr fix
    dst_addr = dst_addr + dst_roi_y * dst_stride + dst_roi_x * dst_channel;

    // use float to keep accuracy
    // float src_dst_scale_h = float(src_roi_h) / dst_roi_h;
    int src_dst_scale_h_int    = src_roi_h / dst_roi_h;
    float src_dst_scale_h_frac = float(src_roi_h - src_dst_scale_h_int * dst_roi_h) / dst_roi_h;

    float dst_src_scale_w = float(dst_roi_w) / src_roi_w;

    int mult = (int)(src_roi_w < dst_roi_w) *
                   (__float2int_up(1.5 * float(dst_roi_w) / float(src_roi_w) + 0.5)) +
               (int)(src_roi_w >= dst_roi_w);
    float dst_pos_w_offset = (dst_src_scale_w * 0.5) - 0.5;
    // float src_pos_h_offset = (src_dst_scale_h * 0.5) - 0.5;
    float src_pos_h_offset = (src_dst_scale_h_int * 0.5 + src_dst_scale_h_frac * 0.5) - 0.5;

    // load copy mult filter
    loadCopyFilter(copy_filter_wram, copy_filter_addr, mult);

    /*############################################*/
    /*#[Module]: NRAM REMAP and w partition      #*/
    /*############################################*/
    // find w nram max load num
    int max_load_w  = nram_find_limit(src_roi_w_, src_roi_w, dst_roi_w, mult, dst_channel) + 2;
    int max_store_w = __float2int_up(max_load_w * dst_src_scale_w);

    // rgba should align to 2 ct lines
    int max_load_w_pad =
        CEIL_ALIGN(max_load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
    int max_store_rgbx_pad = CEIL_ALIGN(max_store_w * dst_channel, HALF_PAD_SIZE);

    // left or right mask and weight
    int max_mask_num   = CEIL_ALIGN(mult * (max_load_w + 1) * kNumDefaultChannel, HALF_PAD_SIZE);
    int max_weight_num = max_store_rgbx_pad;

    // Memory usage
    // 2line y or 2 line uv should align to kNumFilterChIn(conv inst require)
    int max_2line_yuv_pad = CEIL_ALIGN(max_load_w_pad * 2, kNumFilterChIn);

    // origin yuv input(2 line y and 2 line uv) align to UINT8_PAD_SIZE
    int max_input_yuv_pad = CEIL_ALIGN(2 * max_2line_yuv_pad, UINT8_PAD_SIZE);

    // 2line rgba input
    int max_input_rgba = 2 * max_load_w_pad * kNumDefaultChannel;

    // 1 line rgba input after mult
    int max_input_mult_rgba = BANG_MAX(mult * max_load_w_pad * kNumDefaultChannel, max_mask_num);

    // 1 line left and right interp pixel
    int max_4pixel_interp_rgba = 2 * max_store_rgbx_pad;

    // ram1 and ram2 for reuse NRAM
    int ram1_num = max_input_yuv_pad;
    int ram2_num = max_input_rgba;

    if (mult == 1) {
      // origin yuv input(2 line y and 2 line uv) align to UINT8_PAD_SIZE
      // or 1 line left and right interp pixel
      ram1_num = BANG_MAX(max_4pixel_interp_rgba, ram1_num);

      // 2line rgba input
      ram2_num = max_input_rgba;
    } else {
      // origin yuv input(2 line y and 2 line uv) align to UINT8_PAD_SIZE
      // or 1 line rgba input after mult
      ram1_num = BANG_MAX(max_input_mult_rgba, ram1_num);

      // 2line rgba input
      // or 1 line left and right interp pixel
      ram2_num = BANG_MAX(max_4pixel_interp_rgba, ram2_num);
    }

    // Memory NRAM usage
    half *mask_nram   = (half *)compute_temp_nram;
    half *weight_nram = (half *)mask_nram + 2 * max_mask_num;
    half *ram1_nram   = (half *)weight_nram + max_weight_num;
    half *ram2_nram   = (half *)ram1_nram + ram1_num;

    // nram left mask and right mask
    half *mask_left_nram  = mask_nram;
    half *mask_right_nram = mask_left_nram + max_mask_num;

    // divide src_roi_w_ to load_w_repeat * max_load_w + load_w_remaind
    int load_w_repeat = (src_roi_w_ - 2) / (max_load_w - 2);
    int load_w_remain = (src_roi_w_ - 2) % (max_load_w - 2);
    if (load_w_repeat != 0 && load_w_remain == 0) {
      load_w_repeat -= 1;
      load_w_remain = max_load_w - 2;
    }

    if (mult > 1) {
      // pipeline mode
      resizeConvertPipelineMode(
          max_load_w, max_store_w, dst_channel, src_roi_x, src_roi_x_, src_roi_w, src_roi_y,
          src_stride, dst_stride, src_roi_h, dst_roi_h, dst_roi_w, max_mask_num, max_weight_num,
          max_2line_yuv_pad, mult, load_w_repeat, load_w_remain, src_dst_scale_h_int,
          src_dst_scale_h_frac, dst_src_scale_w, dst_pos_w_offset, src_pos_h_offset,
          (uint8_t *)sram_buffer, (int *)sync_buffer, (uint8_t *)src_y_addr, (uint8_t *)src_uv_addr,
          (uint8_t *)dst_addr, (int16_t *)yuv_filter_wram, (half *)yuv_bias_nram,
          (int8_t *)copy_filter_wram, (half *)mask_nram, (half *)weight_nram, (half *)ram1_nram,
          (half *)ram2_nram, (half *)mask_left_nram, (half *)mask_right_nram,
          (half *)mask_left_addr, (half *)mask_right_addr, (half *)weight_right_addr,
          (half *)round_nram, (half *)max_value_nram, (half *)cvt_temp_nram);
    } else {
      // no pipeline mode
      resizeConvertNoPipelineMode(
          max_load_w, max_store_w, dst_channel, src_roi_x, src_roi_x_, src_roi_w, src_roi_y,
          src_stride, dst_stride, src_roi_h, dst_roi_h, dst_roi_w, max_mask_num, max_weight_num,
          max_2line_yuv_pad, mult, load_w_repeat, load_w_remain, src_dst_scale_h_int,
          src_dst_scale_h_frac, dst_src_scale_w, dst_pos_w_offset, src_pos_h_offset,
          (uint8_t *)src_y_addr, (uint8_t *)src_uv_addr, (uint8_t *)dst_addr,
          (int16_t *)yuv_filter_wram, (half *)yuv_bias_nram, (int8_t *)copy_filter_wram,
          (half *)mask_nram, (half *)weight_nram, (half *)ram1_nram, (half *)ram2_nram,
          (half *)mask_left_nram, (half *)mask_right_nram, (half *)mask_left_addr,
          (half *)mask_right_addr, (half *)weight_right_addr, (half *)round_nram,
          (half *)max_value_nram, (half *)cvt_temp_nram);
    }  // mult = 1
  }    // for batch
}
#else
__mlu_global__ void MLUUnion1KernelResizeConvert200(void **src_y_gdram,
                                                    void **src_uv_gdram,
                                                    void **dst_gdram,
                                                    void *fill_color_gdram,
                                                    uint32_t batch_size,
                                                    uint32_t dst_channel,
                                                    void *convert_filter_gdram,
                                                    void *convert_bias_gdram,
                                                    void *shape_gdram,
                                                    void **mask_gdram,
                                                    void **weight_gdram,
                                                    void **copy_filter_gdram) {}
#endif

#endif  // PLUGIN_RESIZE_YUV_TO_RGBA_MLU200_KERNEL_H_
