/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_RESIZE_YUV_TO_RGBA_MLU300_KERNEL_H_
#define PLUGIN_RESIZE_YUV_TO_RGBA_MLU300_KERNEL_H_
#include "plugin_resize_yuv_to_rgba_macro.h"
#if __BANG_ARCH__ >= 322
#define Y0_IDX 2
#define Y1_IDX 3
#define UV0_IDX 4
#define UV1_IDX 5
#define LOAD_MV_TILING_PAD 64
#define STORE_MV_TILING_PAD 32

/*------------------------------ HELP FUNCTIONS ------------------------------*/
// load convert filter and bias from gdram
__mlu_func__ void loadCvtFilter(half *yuv_filter_wram,
                                half *yuv_bias_nram,
                                half *yuv_filter_sram,
                                half *yuv_bias_sram,
                                half *yuv_filter_gdram,
                                half *yuv_bias_gdram) {
  // load filter and bias for yuv2rgba conv
  int32_t yuv_bias_size   = kNumFilterChOut * sizeof(half);
  int32_t yuv_filter_size = kNumFilterHeightIn * kNumFilterChIn * kNumFilterChOut * sizeof(half);
  if (coreId == 0x80) {
    __memcpy_async(yuv_bias_sram, yuv_bias_gdram, yuv_bias_size, GDRAM2SRAM);
    __memcpy_async(yuv_filter_sram, yuv_filter_gdram, yuv_filter_size, GDRAM2SRAM);
  }
  __sync_cluster();
  if (coreId != 0x80) {
    __memcpy_async(yuv_bias_nram, yuv_bias_sram, yuv_bias_size, SRAM2NRAM);
    __memcpy_async(yuv_filter_wram, yuv_filter_sram, yuv_filter_size, SRAM2WRAM);
  }
}

// load mult copy filter from gdram if needed
// mult: the multipiler used in upscaling mode
__mlu_func__ void loadCopyFilter(int8_t *copy_filter_wram,
                                 int8_t *copy_filter_sram,
                                 int8_t *copy_filter_gdram,
                                 const int32_t mult) {
  int32_t copy_filter_size = kNumLt * kNumLt * mult * sizeof(int8_t);
  __sync_cluster();
  if (coreId == 0x80) {
    __memcpy_async(copy_filter_sram, copy_filter_gdram, copy_filter_size, GDRAM2SRAM);
  }
  __sync_cluster();
  if (coreId != 0x80) {
    __memcpy_async(copy_filter_wram, copy_filter_sram, copy_filter_size, SRAM2WRAM);
  }
}

// load select point mask from gdram
__mlu_func__ void loadSelectMask(half *mask_left_nram,
                                 half *mask_right_nram,
                                 half *mask_left_sram,
                                 half *mask_right_sram,
                                 half *mask_left_gdram,
                                 half *mask_right_gdram,
                                 const int32_t load_mask_num,
                                 const int32_t mult) {
  int32_t load_mask_size = load_mask_num * sizeof(half);
  __sync_cluster();
  if (coreId == 0x80) {
    __memcpy_async(mask_left_sram, mask_left_gdram, load_mask_size, GDRAM2SRAM);
    __memcpy_async(mask_right_sram, mask_right_gdram, load_mask_size, GDRAM2SRAM);
  }
  __sync_cluster();
  if (coreId != 0x80) {
    __memcpy_async(mask_left_nram, mask_left_sram, load_mask_size, SRAM2NRAM);
    __memcpy_async(mask_right_nram + mult * kNumDefaultChannel, mask_right_sram, load_mask_size,
                   SRAM2NRAM);
  }
}

// load horizontal weight from gdram
__mlu_func__ void loadWeightValue(half *weight_nram,
                                  half *weight_sram,
                                  half *weight_gdram,
                                  const int32_t load_weight_num) {
  int32_t load_weight_size = load_weight_num * sizeof(half);
  __sync_cluster();
  if (coreId == 0x80) {
    __memcpy_async(weight_sram, weight_gdram, load_weight_size, GDRAM2SRAM);
  }
  __sync_cluster();
  if (coreId != 0x80) {
    __memcpy_async(weight_nram, weight_sram, load_weight_size, SRAM2NRAM);
  }
  __sync_cluster();
}

// find w nram max load num
// mult: the multipiler used in upscaling mode
__mlu_func__ int32_t nram_find_limit(const int32_t src_roi_w_,
                                     const int32_t dst_channel,
                                     const int32_t mult,
                                     const float dst_src_scale_w) {
  /*
                                      NRAM FIGURE
          _________________________________________________________________
          |______|_______|_______|_________|___________|___________|______|
            bias   lmask   rmask   rweight   ram1_ping   ram1_pong   ram2

                                      RAM1 FIGURE
     ________________________________________________________________________________________
     |                                                                              |
     |                       SCALAR_ON_NRAM(DDR_ALIGN_SIZE)                         |
     |                                                                              |
     |____float_____|____float_____|__int32__|__int32__|__int32___|__int32___|______|__half__
       h_weight_cur   h_weight_pre  Y0_OFFSET Y1_OFFSET UV0_OFFSET UV1_OFFSET UNUSED   DATA
      (h_weight_pre) (h_weight_cur)

                                    DATA MOVEMENT
         SHRINK MODE(mult == 1):
         input_yuv(ram1) ------> 2 lines input rgba(ram2) --------> 1 line input rgba(ram2)
                          conv                             interp
         ---------> 2 lines select rgba(ram1) --------> output_rgba(ram1)
          collect                              interp

         EXPAND MODE(mult > 1):
         input_yuv(ram1) ------> 2 lines input rgba(ram2) --------> 1 line input rgba(ram2)
                          conv                             interp
         --------> 1 line input rgba after mult(ram1) ---------> 2 lines select rgba(ram2)
           conv                                        collect
            or
          memcpy
         --------> output_rgba(ram1)
          interp
  */
  int32_t const_size       = kNumFilterChOut * sizeof(half);  // yuv2rgba bias
  int32_t nram_remain_size = NRAM_SIZE - const_size;
  // nram space to store temporary scalar value.
  int32_t scalar_on_nram_size = DDR_ALIGN_SIZE;
  int32_t lower_bound         = 1;
  int32_t upper_bound         = src_roi_w_ / 2 - 1;

  int32_t limit = BANG_MAX(upper_bound, 1);

  if (mult == 1) {
    while (lower_bound < upper_bound - 1) {
      int32_t load_w  = (limit + 1) * 2;
      int32_t store_w = __float2int_up((float)load_w * dst_src_scale_w);

      // rgba should align to 2 ct lines
      int32_t load_w_pad =
          CEIL_ALIGN(load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
      int32_t store_rgba_pad = CEIL_ALIGN(store_w * dst_channel, HALF_PAD_SIZE);

      // 2 point(left + right) mask size
      int32_t mask_pad  = CEIL_ALIGN((load_w + 1) * kNumDefaultChannel, HALF_PAD_SIZE);
      int32_t mask_size = 2 * mask_pad * sizeof(half);

      // 1 point(left) weight_size
      int32_t weight_size = store_rgba_pad * sizeof(half);

      // 2line y or 2 line uv should align to kNumFilterChIn
      int32_t yuv_2line_pad = CEIL_ALIGN(load_w_pad * 2, kNumFilterChIn);
      // add HALF_PAD_SIZE to avoid memory trampling
      int32_t input_2line_pad = 2 * yuv_2line_pad + HALF_PAD_SIZE;

      // 2 line input rgba
      int32_t input_rgba = 2 * load_w_pad * kNumDefaultChannel;

      // 1 line left and right interp pixel
      int32_t interp_2pixel_rgba = 2 * store_rgba_pad;

      // origin yuv input(2 line y and 2 line uv) align to UINT8_PAD_SIZE
      // or left and right interp rgba pixel
      int32_t ram1_size =
          BANG_MAX(input_2line_pad, interp_2pixel_rgba) * sizeof(half) + scalar_on_nram_size;

      // rgba input(2 line)
      int32_t ram2_size = input_rgba * sizeof(half);

      // 2 : ping and pong
      int32_t nram_malloc_size = mask_size + weight_size + 2 * ram1_size + ram2_size;

      if (nram_malloc_size <= nram_remain_size)
        lower_bound = limit;
      else
        upper_bound = limit;

      limit = (lower_bound + upper_bound) / 2;
    }
  } else {  // mult != 1
    while (lower_bound < upper_bound - 1) {
      int32_t load_w  = (limit + 1) * 2;
      int32_t store_w = __float2int_up((float)load_w * dst_src_scale_w);

      // rgba should align to 2 ct lines
      int32_t load_w_pad =
          CEIL_ALIGN(load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
      int32_t store_rgba_pad = CEIL_ALIGN(store_w * dst_channel, HALF_PAD_SIZE);

      // 2 point(left + right) mask size
      int32_t mask_pad  = CEIL_ALIGN(mult * (load_w + 1) * kNumDefaultChannel, HALF_PAD_SIZE);
      int32_t mask_size = 2 * mask_pad * sizeof(half);

      // 1 point(left) weight_size
      int32_t weight_size = store_rgba_pad * sizeof(half);

      // 2line y or 2 line uv should align to kNumFilterChIn
      int32_t yuv_2line_pad = CEIL_ALIGN(load_w_pad * 2, kNumFilterChIn);
      // add HALF_PAD_SIZE to avoid memory trampling
      int32_t input_2line_pad = 2 * yuv_2line_pad + HALF_PAD_SIZE;

      int32_t input_rgba = 2 * load_w_pad * kNumDefaultChannel;
      // 1line rgba after mult
      int32_t input_mult_rgba = mult * load_w_pad * kNumDefaultChannel;
      input_mult_rgba         = BANG_MAX(input_mult_rgba, mask_pad);

      int32_t interp_2pixel_rgba = 2 * store_rgba_pad;

      // origin yuv input(2 line y and 2 line uv) align to UINT8_PAD_SIZE
      // or 1 line rgba input after mult
      // or 1 line output align to HALF_PAD_SIZE
      int32_t ram1_size =
          BANG_MAX(store_rgba_pad, BANG_MAX(input_2line_pad, input_mult_rgba)) * sizeof(half) +
          scalar_on_nram_size;

      // rgba input(2 line) or left and right interp pixel
      int32_t ram2_size = BANG_MAX(interp_2pixel_rgba, input_rgba) * sizeof(half);

      // 2 : ping and pong
      int32_t nram_malloc_size = mask_size + weight_size + 2 * ram1_size + ram2_size;

      if (nram_malloc_size <= nram_remain_size)
        lower_bound = limit;
      else
        upper_bound = limit;

      limit = (lower_bound + upper_bound) / 2;
    }
  }  // mult != 1
  return limit * 2;
}

__mlu_func__ void fusionConvAndRelun(half *rgba_nram,
                                     int16_t *yuv_nram,
                                     int16_t *filter_wram,
                                     half *bias_nram,
                                     const int32_t yuv_2line_pad) {
  __asm__ volatile(
      "conv.nram.f16.fix16.fix16 [%[dst]], [%[src]], [%[filter]],\n\t"
      "%[ci], 2, %[wi], 2, 1, 1, 1, %[co], -20,\n\t"
      ".add.bias([%[bias]]), .relun(255);\n\t"
      :
      : [dst] "r"(rgba_nram), [src] "r"(yuv_nram), [filter] "r"(filter_wram),
        [ci] "r"(kNumFilterChIn), [wi] "r"(yuv_2line_pad / kNumFilterChIn),
        [co] "r"(kNumFilterChOut), [bias] "r"(bias_nram));
}

/*             vector weight mode
   formula:dst = weight * (src1 - src0) + src0
 */
__mlu_func__ void interpByFusionOp(half *dst_nram,
                                   half *src0_nram,
                                   half *src1_nram,
                                   half *weight_nram,
                                   const int32_t elem_num) {
  __bang_sub(src1_nram, src1_nram, src0_nram, elem_num);
  __bang_fusion(FUSION_FMA, dst_nram, src1_nram, weight_nram, src0_nram, elem_num, elem_num);
}

/*             scalar weight mode
    formula:dst = weight * (src1 - src0) + src0
 */
__mlu_func__ void interpByFusionOp(half *dst_nram,
                                   half *src0_nram,
                                   half *src1_nram,
                                   half weight_scalar,
                                   const int32_t elem_num) {
  __bang_sub(src1_nram, src1_nram, src0_nram, elem_num);
  __bang_fusion(FUSION_FMA, dst_nram, src1_nram, weight_scalar, src0_nram, elem_num, elem_num);
}

__mlu_func__ void loadInputFromGDRAM2SRAM(uint8_t *input_y_sram,
                                          uint8_t *input_uv_sram,
                                          uint8_t *cur_y_addr,
                                          uint8_t *cur_uv_addr,
                                          int32_t *scalar_nram,
                                          int32_t &dst_idx_h_start,
                                          int32_t &dst_idx_h_end,
                                          const int32_t weight_idx,
                                          const int32_t store_h_num,
                                          const int32_t load_w,
                                          const int32_t load_size,
                                          const int32_t src_roi_y,
                                          const int32_t src_roi_h,
                                          const int32_t src_stride,
                                          const float src_pos_h_offset,
                                          const float src_dst_scale_h) {
  dst_idx_h_end = dst_idx_h_start + store_h_num - 1;
  if (src_dst_scale_h < 2) {
    int32_t src_idx_h_start = __float2int_dn(dst_idx_h_start * src_dst_scale_h + src_pos_h_offset);
    src_idx_h_start         = BANG_MAX(src_idx_h_start, 0);
    src_idx_h_start         = BANG_MIN(src_idx_h_start, src_roi_h - 1);
    int32_t src_idx_h_end  = __float2int_dn(dst_idx_h_end * src_dst_scale_h + src_pos_h_offset) + 1;
    src_idx_h_end          = BANG_MIN(src_idx_h_end, src_roi_h - 1);
    int32_t src_y_h_start  = src_idx_h_start + src_roi_y;
    int32_t src_y_h_end    = src_idx_h_end + src_roi_y;
    int32_t src_uv_h_start = src_y_h_start / 2;
    int32_t src_uv_h_end   = src_y_h_end / 2;
    int32_t load_y_h       = src_y_h_end - src_y_h_start + 1;
    int32_t load_uv_h      = src_uv_h_end - src_uv_h_start + 1;
    __memcpy_async(input_y_sram, cur_y_addr + src_y_h_start * src_stride, load_size, GDRAM2SRAM,
                   load_size, src_stride, load_y_h - 1);
    __memcpy_async(input_uv_sram, cur_uv_addr + src_uv_h_start * src_stride, load_size, GDRAM2SRAM,
                   load_size, src_stride, load_uv_h - 1);
    if (coreId != 0x80 && coreId < store_h_num) {
      int32_t dst_idx_h_cur              = dst_idx_h_start + coreId;
      float src_idx_h_cur                = dst_idx_h_cur * src_dst_scale_h + src_pos_h_offset;
      src_idx_h_cur                      = BANG_MAX(src_idx_h_cur, 0);
      src_idx_h_cur                      = BANG_MIN(src_idx_h_cur, src_roi_h - 1);
      int32_t src_idx_h_int              = __float2int_dn(src_idx_h_cur);
      float h_weight_cur                 = src_idx_h_cur - (float)src_idx_h_int;
      int32_t src_y_idx_h0               = src_idx_h_int + src_roi_y;
      int32_t src_uv_idx_h0              = src_y_idx_h0 / 2;
      src_idx_h_int                      = BANG_MIN(src_idx_h_int + 1, src_roi_h - 1);
      int32_t src_y_idx_h1               = src_idx_h_int + src_roi_y;
      int32_t src_uv_idx_h1              = src_y_idx_h1 / 2;
      int32_t sram_y_h0_offset           = (src_y_idx_h0 - src_y_h_start) * load_w;
      int32_t sram_y_h1_offset           = (src_y_idx_h1 - src_y_h_start) * load_w;
      int32_t sram_uv_h0_offset          = (src_uv_idx_h0 - src_uv_h_start) * load_w;
      int32_t sram_uv_h1_offset          = (src_uv_idx_h1 - src_uv_h_start) * load_w;
      ((float *)scalar_nram)[weight_idx] = h_weight_cur;
      scalar_nram[Y0_IDX]                = sram_y_h0_offset;
      scalar_nram[Y1_IDX]                = sram_y_h1_offset;
      scalar_nram[UV0_IDX]               = sram_uv_h0_offset;
      scalar_nram[UV1_IDX]               = sram_uv_h1_offset;
    }
  } else {
    for (int32_t store_h_iter = 0; store_h_iter < store_h_num; ++store_h_iter) {
      int32_t dst_idx_h_cur = dst_idx_h_start + store_h_iter;
      float src_idx_h_cur   = dst_idx_h_cur * src_dst_scale_h + src_pos_h_offset;
      src_idx_h_cur         = BANG_MIN(src_idx_h_cur, src_roi_h - 1);
      int32_t src_idx_h_int = __float2int_dn(src_idx_h_cur);
      float h_weight_cur    = src_idx_h_cur - (float)src_idx_h_int;
      int32_t src_y_idx_h0  = src_idx_h_int + src_roi_y;
      int32_t src_uv_idx_h0 = src_y_idx_h0 / 2;
      src_idx_h_int         = BANG_MIN(src_idx_h_int + 1, src_roi_h - 1);
      int32_t src_y_idx_h1  = src_idx_h_int + src_roi_y;
      int32_t src_uv_idx_h1 = src_y_idx_h1 / 2;
      __memcpy_async(input_y_sram + store_h_iter * 2 * load_w,
                     cur_y_addr + src_y_idx_h0 * src_stride, load_size, GDRAM2SRAM);
      __memcpy_async(input_uv_sram + store_h_iter * 2 * load_w,
                     cur_uv_addr + src_uv_idx_h0 * src_stride, load_size, GDRAM2SRAM);
      __memcpy_async(input_y_sram + (store_h_iter * 2 + 1) * load_w,
                     cur_y_addr + src_y_idx_h1 * src_stride, load_size, GDRAM2SRAM);
      __memcpy_async(input_uv_sram + (store_h_iter * 2 + 1) * load_w,
                     cur_uv_addr + src_uv_idx_h1 * src_stride, load_size, GDRAM2SRAM);
      if (store_h_iter == coreId) {
        ((float *)scalar_nram)[weight_idx] = h_weight_cur;
        scalar_nram[Y0_IDX]                = (coreId * 2) * load_w;
        scalar_nram[Y1_IDX]                = (coreId * 2 + 1) * load_w;
        scalar_nram[UV0_IDX]               = (coreId * 2) * load_w;
        scalar_nram[UV1_IDX]               = (coreId * 2 + 1) * load_w;
      }
    }
  }
  dst_idx_h_start = dst_idx_h_end + 1;
}

__mlu_func__ void mvAndCvtInputFromSRAM2NRAM(int16_t *load_y_nram,
                                             int16_t *load_uv_nram,
                                             uint8_t *input_y_sram,
                                             uint8_t *input_uv_sram,
                                             int32_t *scalar_nram,
                                             const int32_t store_h_num,
                                             const int32_t load_w,
                                             const int32_t load_w_pad) {
  if (coreId == 0x80 || coreId >= store_h_num) {
    return;
  }

  int32_t mv_tiling_repeat = CEIL_ALIGN(load_w, LOAD_MV_TILING_PAD) / LOAD_MV_TILING_PAD;
  // load two lines of Y and two lines of UV
  __asm__ volatile(
      "move.tiling.async.nram.sram.b128 [%[dst]], [%[src]],\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 64, 1, 0,\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 128, 1, 0,\n\t"
      ".cvt.fix16.u8(-7);\n\t"
      :
      : [dst] "r"(load_y_nram), [src] "r"(input_y_sram + scalar_nram[Y0_IDX]),
        [n4] "r"(mv_tiling_repeat));
  __asm__ volatile(
      "move.tiling.async.nram.sram.b128 [%[dst]], [%[src]],\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 64, 1, 0,\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 128, 1, 0,\n\t"
      ".cvt.fix16.u8(-7);\n\t"
      :
      : [dst] "r"(load_y_nram + load_w_pad), [src] "r"(input_y_sram + scalar_nram[Y1_IDX]),
        [n4] "r"(mv_tiling_repeat));
  __asm__ volatile(
      "move.tiling.async.nram.sram.b128 [%[dst]], [%[src]],\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 64, 1, 0,\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 128, 1, 0,\n\t"
      ".cvt.fix16.u8(-7);\n\t"
      :
      : [dst] "r"(load_uv_nram), [src] "r"(input_uv_sram + scalar_nram[UV0_IDX]),
        [n4] "r"(mv_tiling_repeat));
  __asm__ volatile(
      "move.tiling.async.nram.sram.b128 [%[dst]], [%[src]],\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 64, 1, 0,\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 128, 1, 0,\n\t"
      ".cvt.fix16.u8(-7);\n\t"
      :
      : [dst] "r"(load_uv_nram + load_w_pad), [src] "r"(input_uv_sram + scalar_nram[UV1_IDX]),
        [n4] "r"(mv_tiling_repeat));
}

__mlu_func__ void computeStage(half *store_rgba_nram,
                               int16_t *load_yuv_nram,
                               int16_t *yuv_filter_wram,
                               int8_t *copy_filter_wram,
                               half *calc_nram,
                               half *yuv_bias_nram,
                               half *mask_left_nram,
                               half *mask_right_nram,
                               half *w_weight_nram,
                               const half h_weight_scalar,
                               const int32_t store_h_num,
                               const int32_t load_w_pad,
                               const int32_t yuv_2line_pad,
                               const int32_t rgba_pad,
                               const int32_t load_mask_pad,
                               const int32_t load_weight_pad,
                               const int32_t mult) {
  if (coreId == 0x80 || coreId >= store_h_num) {
    return;
  }

  // nram reuse
  half *rgba_nram        = calc_nram;
  half *rgba_select_nram = (half *)load_yuv_nram;
  half *rgba_mult_nram   = calc_nram;
  if (mult > 1) {
    rgba_mult_nram   = (half *)load_yuv_nram;
    rgba_select_nram = calc_nram;
  }

  /*######################*/
  /*#[Module]: YUV2RGBX#*/
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
  * [ 1.164  0     0 ... 0] -> 1.164 * Y1,1 -> G:q
  1,1
  * [-0.392 -0.813 0 ... 0]  - 0.392 * U1,1
  *                          - 0.813 * V1,1
  * ...
  * ...
  * [ 0 0 1.164 0     0 ... 0] -> 1.164 * Y1,3 -> R1,3
  * [ 0 0 0     1.586 0 ... 0]  + 1.586 * V1,2
  * ...
  * Total 4 X kNumFilterChIn pixcels hence 4 X kNumFilterChIn kernels
  */
  fusionConvAndRelun(rgba_nram, load_yuv_nram, yuv_filter_wram, yuv_bias_nram, yuv_2line_pad);

  // interpolation top line and bottom line
  half *rgba_top_line_nram = rgba_nram;
  half *rgba_bot_line_nram = rgba_top_line_nram + rgba_pad;
  interpByFusionOp(rgba_nram, rgba_top_line_nram, rgba_bot_line_nram, h_weight_scalar, rgba_pad);

  /*#################################*/
  /*#[Module]: src image expansion#*/
  /*#################################*/
  if (mult > 1 && mult <= kNumMultLimit) {
    __bang_conv((int16_t *)rgba_mult_nram, (int16_t *)rgba_nram, (int8_t *)copy_filter_wram, kNumLt,
                1, rgba_pad / kNumLt, 1, 1, 1, 1, kNumLt * mult, 0);
  } else if (mult > kNumMultLimit) {
    __memcpy((half *)rgba_mult_nram, (half *)rgba_nram,
             kNumDefaultChannel * sizeof(half),         // size
             NRAM2NRAM,                                 // direction
             kNumDefaultChannel * sizeof(half),         // dst_stride0
             mult - 1,                                  // dst_seg0
             mult * kNumDefaultChannel * sizeof(half),  // dst_stride1
             load_w_pad - 1,                            // dst_seg1
             0,                                         // src_stride0
             mult - 1,                                  // src_seg0
             kNumDefaultChannel * sizeof(half),         // src_stride1
             load_w_pad - 1);                           // src_seg1
  }

  // Select data using the mask generated
  half *select_left_line_nram  = rgba_select_nram;
  half *select_right_line_nram = select_left_line_nram + load_weight_pad;
  // set border of select right to zero to make calculation correctly
  __bang_write_zero(select_right_line_nram, load_weight_pad);
  __bang_collect((half *)select_left_line_nram, (half *)rgba_mult_nram, (half *)mask_left_nram,
                 load_mask_pad);  // left points
  __bang_collect((half *)select_right_line_nram, (half *)rgba_mult_nram, (half *)mask_right_nram,
                 load_mask_pad);  // right points

  // interpolation left line and right line
  interpByFusionOp(store_rgba_nram, select_left_line_nram, select_right_line_nram, w_weight_nram,
                   load_weight_pad);
}

__mlu_func__ void mvAndCvtOutputFromNRAM2SRAM(uint8_t *output_sram,
                                              half *store_rgba_nram,
                                              const int32_t store_h_num,
                                              const int32_t store_num_pad) {
  if (coreId == 0x80 || coreId >= store_h_num) {
    return;
  }

  int32_t mv_tiling_repeat = store_num_pad / STORE_MV_TILING_PAD;
  __asm__ volatile(
      "move.tiling.async.sram.nram.b128 [%[dst]], [%[src]],\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 64, 1, 0,\n\t"
      "64, 1, 0, 1, 0, 1, 0, %[n4], 32, 1, 0,\n\t"
      ".cvt.rn.u8.f16();"
      :
      : [dst] "r"(output_sram + coreId * store_num_pad), [src] "r"(store_rgba_nram),
        [n4] "r"(mv_tiling_repeat));
}

__mlu_func__ void storeOutputFromSRAM2GDRAM(uint8_t *dst_addr,
                                            uint8_t *output_sram,
                                            const int32_t dst_stride,
                                            const int32_t src_stride,
                                            const int32_t store_size,
                                            const int32_t store_h_num) {
  if (coreId == 0x80) {
    __memcpy_async(dst_addr, output_sram, store_size, SRAM2GDRAM, dst_stride, src_stride,
                   store_h_num - 1);
  }
}

__mlu_func__ void resizeConvert5PipelineMode(uint8_t *dst_addr,
                                             uint8_t *src_y_addr,
                                             uint8_t *src_uv_addr,
                                             int8_t *copy_filter_addr,
                                             half *mask_left_addr,
                                             half *mask_right_addr,
                                             half *weight_right_addr,
                                             uint8_t *sram_buffer,
                                             half *yuv_bias_nram,
                                             half *compute_temp_nram,
                                             int16_t *yuv_filter_wram,
                                             int8_t *copy_filter_wram,
                                             uint32_t src_stride,
                                             uint32_t src_roi_x,
                                             uint32_t src_roi_y,
                                             uint32_t src_roi_w,
                                             uint32_t src_roi_h,
                                             uint32_t dst_stride,
                                             uint32_t dst_roi_w,
                                             uint32_t dst_roi_h,
                                             uint32_t dst_channel,
                                             int32_t h_cluster_num,
                                             int32_t h_cluster_start) {
  int32_t src_roi_x_ = src_roi_x / 2 * 2;
  int32_t src_roi_w_ = (src_roi_x % 2 + src_roi_w + 1) / 2 * 2;
  // use float to keep accuracy
  float dst_src_scale_w  = float(dst_roi_w) / src_roi_w;
  float src_dst_scale_w  = float(src_roi_w) / dst_roi_w;
  float src_dst_scale_h  = float(src_roi_h) / dst_roi_h;
  float dst_pos_w_offset = (dst_src_scale_w * 0.5) - 0.5;
  float src_pos_w_offset = (src_dst_scale_w * 0.5) - 0.5;
  float src_pos_h_offset = (src_dst_scale_h * 0.5) - 0.5;
  // src point need be copyed mult times
  int32_t mult = (int32_t)(src_roi_w < dst_roi_w) * (__float2int_up(1.5 * dst_src_scale_w + 0.5)) +
                 (int32_t)(src_roi_w >= dst_roi_w);

  int32_t dst_idx_h_start = h_cluster_start;
  int32_t dst_idx_h_end   = 0;
  int32_t dst_idx_w_start = 0;
  int32_t dst_idx_w_end   = 0;

  int32_t src_idx_w     = 0 - (src_roi_x % 2);
  int32_t src_idx_w_end = src_idx_w + src_roi_w_;

  /*#############################*/
  /*#[Module]: w partition      #*/
  /*#############################*/
  // find w max load num
  int32_t max_load_w  = nram_find_limit(src_roi_w_, dst_channel, mult, dst_src_scale_w) + 2;
  int32_t max_store_w = __float2int_up(max_load_w * dst_src_scale_w);

  // rgba align to 2 ct lines
  int32_t max_load_w_pad =
      CEIL_ALIGN(max_load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
  int32_t max_store_rgba_pad = CEIL_ALIGN(max_store_w * dst_channel, HALF_PAD_SIZE);

  // mask zone and weight zone
  int32_t max_mask_num   = CEIL_ALIGN(mult * (max_load_w + 1) * kNumDefaultChannel, HALF_PAD_SIZE);
  int32_t max_weight_num = max_store_rgba_pad;

  // 2 line y or 2 line uv should align to kNumFilterChIn(conv inst require)
  int32_t max_2line_yuv_pad = CEIL_ALIGN(max_load_w_pad * 2, kNumFilterChIn);
  // add HALF_PAD_SIZE to avoid memory trampling
  int32_t max_2line_input_pad = 2 * max_2line_yuv_pad + HALF_PAD_SIZE;

  // 1 line rgba input after mult
  int32_t max_input_mult_rgba = max_load_w_pad * kNumDefaultChannel * mult;
  max_input_mult_rgba         = BANG_MAX(max_input_mult_rgba, max_mask_num);

  // 1 line left and right interp pixel
  int32_t max_2pixel_interp_rgba = 2 * max_store_rgba_pad;

  // data zone: store src data or dst data and temporary result
  // some scalar store on nram
  int32_t data_zone_size      = max_2line_input_pad;
  int32_t scalar_on_nram_size = DDR_ALIGN_SIZE;

  if (mult == 1) {
    // origin yuv input (2 line y and 2 line uv)
    // or left and right interp pixel
    data_zone_size =
        BANG_MAX(max_2pixel_interp_rgba, data_zone_size) * sizeof(half) + scalar_on_nram_size;
  } else {
    // origin yuv input (2 line y and 2 line uv)
    // or 1 line rgba input after mult
    // or output align to HALF_PAD_SIZE
    data_zone_size =
        BANG_MAX(BANG_MAX(max_input_mult_rgba, max_store_rgba_pad), data_zone_size) * sizeof(half) +
        scalar_on_nram_size;
  }
  // Memory NRAM usage
  half *mask_left_nram    = (half *)compute_temp_nram;
  half *mask_right_nram   = mask_left_nram + max_mask_num;
  half *weight_nram       = mask_right_nram + max_mask_num;
  uint8_t *data_ping_nram = (uint8_t *)(weight_nram + max_weight_num);
  half *calc_nram         = (half *)(data_ping_nram + 2 * data_zone_size);
  int32_t nram_pong       = data_zone_size;

  /*#############################*/
  /*#[Module]: h partition      #*/
  /*#############################*/
  int32_t max_store_h = coreDim;
  int32_t max_load_h  = max_store_h * 2;

  int32_t max_input_y  = max_load_h * max_load_w;
  int32_t max_input_uv = max_load_h * max_load_w;

  if (src_dst_scale_h < 2) {
    max_load_h   = __float2int_up((max_store_h - 1) * src_dst_scale_h) + 2;
    max_input_y  = max_load_h * max_load_w;
    max_input_uv = (max_load_h / 2 + 1) * max_load_w;
  }

  // Memory SRAM usage
  uint8_t *input_y_ping_sram  = (uint8_t *)sram_buffer;
  uint8_t *input_uv_ping_sram = input_y_ping_sram + max_input_y;
  uint8_t *output_ping_sram   = input_uv_ping_sram + max_input_uv;

  int32_t store_h_repeat = h_cluster_num / max_store_h;
  int32_t store_h_remain = h_cluster_num % max_store_h;

  // load copy filter if need src expansion and can be expanded through convolution
  if (mult > 1 && mult <= kNumMultLimit) {
    int8_t *copy_filter_sram = (int8_t *)sram_buffer;
    loadCopyFilter(copy_filter_wram, copy_filter_sram, copy_filter_addr, mult);
  }

  // cur roi y and uv addr
  uint8_t *cur_y_addr  = src_y_addr + src_roi_x_;
  uint8_t *cur_uv_addr = src_uv_addr + src_roi_x_;

  while (src_idx_w < src_idx_w_end) {
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
    int32_t load_w_remain = src_idx_w_end - src_idx_w;
    int32_t load_w        = max_load_w;
    int32_t compute_w     = max_load_w - 2;
    if (load_w_remain <= max_load_w) {
      load_w    = load_w_remain;
      compute_w = load_w;
    } else {
      int32_t cur_src_idx_w = src_idx_w + compute_w;
      int32_t cur_dst_idx_w =
          __float2int_up((float)cur_src_idx_w * dst_src_scale_w + dst_pos_w_offset);
      cur_dst_idx_w = BANG_MIN(cur_dst_idx_w, dst_roi_w);
      int32_t related_src_idx_w0 =
          __float2int_dn((float)(cur_dst_idx_w - 1) * src_dst_scale_w + src_pos_w_offset);
      int32_t related_src_idx_w1 =
          __float2int_dn((float)(cur_dst_idx_w)*src_dst_scale_w + src_pos_w_offset);
      while (related_src_idx_w0 >= cur_src_idx_w || related_src_idx_w1 < cur_src_idx_w) {
        load_w -= 2;
        compute_w -= 2;
        cur_src_idx_w = src_idx_w + compute_w;
        cur_dst_idx_w = __float2int_up((float)cur_src_idx_w * dst_src_scale_w + dst_pos_w_offset);
        cur_dst_idx_w = BANG_MIN(cur_dst_idx_w, dst_roi_w);
        related_src_idx_w0 =
            __float2int_dn((float)(cur_dst_idx_w - 1) * src_dst_scale_w + src_pos_w_offset);
        related_src_idx_w1 =
            __float2int_dn((float)(cur_dst_idx_w)*src_dst_scale_w + src_pos_w_offset);
      }
    }

    // update src w idx
    src_idx_w += compute_w;

    // compute dst w idx end
    dst_idx_w_end = __float2int_up((float)src_idx_w * dst_src_scale_w + dst_pos_w_offset);
    dst_idx_w_end = BANG_MIN(dst_idx_w_end, dst_roi_w);

    // compute dst store num
    int32_t store_rgba = (dst_idx_w_end - dst_idx_w_start) * dst_channel;

    // update dst w idx start
    dst_idx_w_start = dst_idx_w_end;

    // mask load num each w_iter
    int32_t load_mask     = mult * kNumDefaultChannel * compute_w;
    int32_t load_mask_pad = CEIL_ALIGN(load_mask + mult * kNumDefaultChannel, HALF_PAD_SIZE);

    // weight load num each w inter
    int32_t load_weight     = store_rgba;
    int32_t load_weight_pad = CEIL_ALIGN(load_weight, HALF_PAD_SIZE);

    if (store_rgba > 0) {
      // load left mask and right mask
      int32_t init_pad = CEIL_ALIGN(mult * kNumDefaultChannel, HALF_PAD_SIZE);
      // first pixel of right mask should be zero
      __bang_write_zero(mask_right_nram, init_pad);
      // set border of mask to zero in case that NAN or INF on this space
      __bang_write_zero(mask_left_nram + load_mask_pad - init_pad, init_pad);
      __bang_write_zero(mask_right_nram + load_mask_pad - HALF_PAD_SIZE, HALF_PAD_SIZE);
      half *mask_left_sram  = (half *)sram_buffer;
      half *mask_right_sram = mask_left_sram + load_mask_pad;
      loadSelectMask(mask_left_nram, mask_right_nram, mask_left_sram, mask_right_sram,
                     mask_left_addr, mask_right_addr, load_mask, mult);

      // load weight
      half *weight_sram = (half *)sram_buffer;
      loadWeightValue(weight_nram, weight_sram, weight_right_addr, load_weight);

      int32_t load_w_pad =
          CEIL_ALIGN(load_w * kNumDefaultChannel, HALF_PAD_SIZE) / kNumDefaultChannel;
      int32_t rgba_pad      = load_w_pad * kNumDefaultChannel;
      int32_t yuv_2line_pad = CEIL_ALIGN(load_w_pad * 2, kNumFilterChIn);
      // datasize of src data
      int32_t load_data_size  = load_w * sizeof(uint8_t);
      int32_t store_data_size = store_rgba * sizeof(uint8_t);
      // Memory NRAM remap
      uint8_t *scalar_ping_nram     = data_ping_nram;
      uint8_t *load_y_ping_nram     = scalar_ping_nram + scalar_on_nram_size;
      uint8_t *load_uv_ping_nram    = load_y_ping_nram + yuv_2line_pad * sizeof(half);
      uint8_t *store_rgba_ping_nram = load_y_ping_nram;

      if (store_h_repeat > 0) {
        loadInputFromGDRAM2SRAM(input_y_ping_sram, input_uv_ping_sram, cur_y_addr, cur_uv_addr,
                                (int32_t *)scalar_ping_nram, dst_idx_h_start, dst_idx_h_end, 0,
                                max_store_h, load_w, load_data_size, src_roi_y, src_roi_h,
                                src_stride, src_pos_h_offset, src_dst_scale_h);
        __sync_cluster();
      }

      if (store_h_repeat > 1) {
        loadInputFromGDRAM2SRAM(input_y_ping_sram + SRAM_PONG, input_uv_ping_sram + SRAM_PONG,
                                cur_y_addr, cur_uv_addr, (int32_t *)(scalar_ping_nram + nram_pong),
                                dst_idx_h_start, dst_idx_h_end, 0, max_store_h, load_w,
                                load_data_size, src_roi_y, src_roi_h, src_stride, src_pos_h_offset,
                                src_dst_scale_h);
        mvAndCvtInputFromSRAM2NRAM((int16_t *)load_y_ping_nram, (int16_t *)load_uv_ping_nram,
                                   input_y_ping_sram, input_uv_ping_sram,
                                   (int32_t *)scalar_ping_nram, max_store_h, load_w, load_w_pad);
        __sync_cluster();
      }

      if (store_h_repeat > 2) {
        loadInputFromGDRAM2SRAM(input_y_ping_sram, input_uv_ping_sram, cur_y_addr, cur_uv_addr,
                                (int32_t *)scalar_ping_nram, dst_idx_h_start, dst_idx_h_end, 1,
                                max_store_h, load_w, load_data_size, src_roi_y, src_roi_h,
                                src_stride, src_pos_h_offset, src_dst_scale_h);
        mvAndCvtInputFromSRAM2NRAM(
            (int16_t *)(load_y_ping_nram + nram_pong), (int16_t *)(load_uv_ping_nram + nram_pong),
            input_y_ping_sram + SRAM_PONG, input_uv_ping_sram + SRAM_PONG,
            (int32_t *)(scalar_ping_nram + nram_pong), max_store_h, load_w, load_w_pad);
        computeStage((half *)store_rgba_ping_nram, (int16_t *)load_y_ping_nram, yuv_filter_wram,
                     copy_filter_wram, calc_nram, yuv_bias_nram, mask_left_nram, mask_right_nram,
                     weight_nram, ((float *)scalar_ping_nram)[0], max_store_h, load_w_pad,
                     yuv_2line_pad, rgba_pad, load_mask_pad, load_weight_pad, mult);
        __sync_cluster();
      }

      if (store_h_repeat > 3) {
        loadInputFromGDRAM2SRAM(input_y_ping_sram + SRAM_PONG, input_uv_ping_sram + SRAM_PONG,
                                cur_y_addr, cur_uv_addr, (int32_t *)(scalar_ping_nram + nram_pong),
                                dst_idx_h_start, dst_idx_h_end, 1, max_store_h, load_w,
                                load_data_size, src_roi_y, src_roi_h, src_stride, src_pos_h_offset,
                                src_dst_scale_h);
        computeStage((half *)(store_rgba_ping_nram + nram_pong),
                     (int16_t *)(load_y_ping_nram + nram_pong), yuv_filter_wram, copy_filter_wram,
                     calc_nram, yuv_bias_nram, mask_left_nram, mask_right_nram, weight_nram,
                     ((float *)(scalar_ping_nram + nram_pong))[0], max_store_h, load_w_pad,
                     yuv_2line_pad, rgba_pad, load_mask_pad, load_weight_pad, mult);
        mvAndCvtOutputFromNRAM2SRAM(output_ping_sram, (half *)store_rgba_ping_nram, max_store_h,
                                    load_weight_pad);
        mvAndCvtInputFromSRAM2NRAM((int16_t *)load_y_ping_nram, (int16_t *)load_uv_ping_nram,
                                   input_y_ping_sram, input_uv_ping_sram,
                                   (int32_t *)scalar_ping_nram, max_store_h, load_w, load_w_pad);
        __sync_cluster();
      }

      for (int32_t i = 0; i < store_h_repeat - 4; ++i) {
        storeOutputFromSRAM2GDRAM(dst_addr + i * max_store_h * dst_stride,
                                  output_ping_sram + (i % 2) * SRAM_PONG, dst_stride,
                                  load_weight_pad, store_data_size, max_store_h);
        loadInputFromGDRAM2SRAM(
            input_y_ping_sram + (i % 2) * SRAM_PONG, input_uv_ping_sram + (i % 2) * SRAM_PONG,
            cur_y_addr, cur_uv_addr, (int32_t *)(scalar_ping_nram + (i % 2) * nram_pong),
            dst_idx_h_start, dst_idx_h_end, (i % 4) / 2, max_store_h, load_w, load_data_size,
            src_roi_y, src_roi_h, src_stride, src_pos_h_offset, src_dst_scale_h);
        computeStage(
            (half *)(store_rgba_ping_nram + (i % 2) * nram_pong),
            (int16_t *)(load_y_ping_nram + (i % 2) * nram_pong), yuv_filter_wram, copy_filter_wram,
            calc_nram, yuv_bias_nram, mask_left_nram, mask_right_nram, weight_nram,
            ((float *)(scalar_ping_nram + (i % 2) * nram_pong))[(i + 2) % 4 / 2], max_store_h,
            load_w_pad, yuv_2line_pad, rgba_pad, load_mask_pad, load_weight_pad, mult);
        mvAndCvtOutputFromNRAM2SRAM(output_ping_sram + ((i + 1) % 2) * SRAM_PONG,
                                    (half *)(store_rgba_ping_nram + ((i + 1) % 2) * nram_pong),
                                    max_store_h, load_weight_pad);
        mvAndCvtInputFromSRAM2NRAM((int16_t *)(load_y_ping_nram + ((i + 1) % 2) * nram_pong),
                                   (int16_t *)(load_uv_ping_nram + ((i + 1) % 2) * nram_pong),
                                   input_y_ping_sram + ((i + 1) % 2) * SRAM_PONG,
                                   input_uv_ping_sram + ((i + 1) % 2) * SRAM_PONG,
                                   (int32_t *)(scalar_ping_nram + ((i + 1) % 2) * nram_pong),
                                   max_store_h, load_w, load_w_pad);
        __sync_cluster();
      }

      if (store_h_repeat > 3) {
        storeOutputFromSRAM2GDRAM(dst_addr + (store_h_repeat - 4) * max_store_h * dst_stride,
                                  output_ping_sram + (store_h_repeat % 2) * SRAM_PONG, dst_stride,
                                  load_weight_pad, store_data_size, max_store_h);
      }
      if (store_h_remain > 0) {
        loadInputFromGDRAM2SRAM(
            input_y_ping_sram + (store_h_repeat % 2) * SRAM_PONG,
            input_uv_ping_sram + (store_h_repeat % 2) * SRAM_PONG, cur_y_addr, cur_uv_addr,
            (int32_t *)(scalar_ping_nram + (store_h_repeat % 2) * nram_pong), dst_idx_h_start,
            dst_idx_h_end, (store_h_repeat % 4) / 2, store_h_remain, load_w, load_data_size,
            src_roi_y, src_roi_h, src_stride, src_pos_h_offset, src_dst_scale_h);
      }
      if (store_h_repeat > 1) {
        computeStage((half *)(store_rgba_ping_nram + (store_h_repeat % 2) * nram_pong),
                     (int16_t *)(load_y_ping_nram + (store_h_repeat % 2) * nram_pong),
                     yuv_filter_wram, copy_filter_wram, calc_nram, yuv_bias_nram, mask_left_nram,
                     mask_right_nram, weight_nram,
                     ((float *)(scalar_ping_nram +
                                (store_h_repeat % 2) * nram_pong))[(store_h_repeat - 2) % 4 / 2],
                     max_store_h, load_w_pad, yuv_2line_pad, rgba_pad, load_mask_pad,
                     load_weight_pad, mult);
      }
      if (store_h_repeat > 2) {
        mvAndCvtOutputFromNRAM2SRAM(
            output_ping_sram + ((store_h_repeat + 1) % 2) * SRAM_PONG,
            (half *)(store_rgba_ping_nram + ((store_h_repeat + 1) % 2) * nram_pong), max_store_h,
            load_weight_pad);
      }
      if (store_h_repeat > 0) {
        mvAndCvtInputFromSRAM2NRAM(
            (int16_t *)(load_y_ping_nram + ((store_h_repeat + 1) % 2) * nram_pong),
            (int16_t *)(load_uv_ping_nram + ((store_h_repeat + 1) % 2) * nram_pong),
            input_y_ping_sram + ((store_h_repeat + 1) % 2) * SRAM_PONG,
            input_uv_ping_sram + ((store_h_repeat + 1) % 2) * SRAM_PONG,
            (int32_t *)(scalar_ping_nram + ((store_h_repeat + 1) % 2) * nram_pong), max_store_h,
            load_w, load_w_pad);
      }
      __sync_cluster();

      if (store_h_repeat > 2) {
        storeOutputFromSRAM2GDRAM(dst_addr + (store_h_repeat - 3) * max_store_h * dst_stride,
                                  output_ping_sram + ((store_h_repeat + 1) % 2) * SRAM_PONG,
                                  dst_stride, load_weight_pad, store_data_size, max_store_h);
      }
      if (store_h_repeat > 0) {
        computeStage((half *)(store_rgba_ping_nram + ((store_h_repeat + 1) % 2) * nram_pong),
                     (int16_t *)(load_y_ping_nram + ((store_h_repeat + 1) % 2) * nram_pong),
                     yuv_filter_wram, copy_filter_wram, calc_nram, yuv_bias_nram, mask_left_nram,
                     mask_right_nram, weight_nram,
                     ((float *)(scalar_ping_nram + ((store_h_repeat + 1) % 2) *
                                                       nram_pong))[(store_h_repeat - 1) % 4 / 2],
                     max_store_h, load_w_pad, yuv_2line_pad, rgba_pad, load_mask_pad,
                     load_weight_pad, mult);
      }
      if (store_h_repeat > 1) {
        mvAndCvtOutputFromNRAM2SRAM(
            output_ping_sram + (store_h_repeat % 2) * SRAM_PONG,
            (half *)(store_rgba_ping_nram + (store_h_repeat % 2) * nram_pong), max_store_h,
            load_weight_pad);
      }
      if (store_h_remain > 0) {
        mvAndCvtInputFromSRAM2NRAM(
            (int16_t *)(load_y_ping_nram + (store_h_repeat % 2) * nram_pong),
            (int16_t *)(load_uv_ping_nram + (store_h_repeat % 2) * nram_pong),
            input_y_ping_sram + (store_h_repeat % 2) * SRAM_PONG,
            input_uv_ping_sram + (store_h_repeat % 2) * SRAM_PONG,
            (int32_t *)(scalar_ping_nram + (store_h_repeat % 2) * nram_pong), store_h_remain,
            load_w, load_w_pad);
      }
      __sync_cluster();

      if (store_h_repeat > 1) {
        storeOutputFromSRAM2GDRAM(dst_addr + (store_h_repeat - 2) * max_store_h * dst_stride,
                                  output_ping_sram + (store_h_repeat % 2) * SRAM_PONG, dst_stride,
                                  load_weight_pad, store_data_size, max_store_h);
      }
      if (store_h_remain > 0) {
        computeStage((half *)(store_rgba_ping_nram + (store_h_repeat % 2) * nram_pong),
                     (int16_t *)(load_y_ping_nram + (store_h_repeat % 2) * nram_pong),
                     yuv_filter_wram, copy_filter_wram, calc_nram, yuv_bias_nram, mask_left_nram,
                     mask_right_nram, weight_nram,
                     ((float *)(scalar_ping_nram +
                                (store_h_repeat % 2) * nram_pong))[store_h_repeat % 4 / 2],
                     store_h_remain, load_w_pad, yuv_2line_pad, rgba_pad, load_mask_pad,
                     load_weight_pad, mult);
      }
      if (store_h_repeat > 0) {
        mvAndCvtOutputFromNRAM2SRAM(
            output_ping_sram + ((store_h_repeat + 1) % 2) * SRAM_PONG,
            (half *)(store_rgba_ping_nram + ((store_h_repeat + 1) % 2) * nram_pong), max_store_h,
            load_weight_pad);
      }
      __sync_cluster();

      if (store_h_repeat > 0) {
        storeOutputFromSRAM2GDRAM(dst_addr + (store_h_repeat - 1) * max_store_h * dst_stride,
                                  output_ping_sram + ((store_h_repeat + 1) % 2) * SRAM_PONG,
                                  dst_stride, load_weight_pad, store_data_size, max_store_h);
      }
      if (store_h_remain > 0) {
        mvAndCvtOutputFromNRAM2SRAM(
            output_ping_sram + (store_h_repeat % 2) * SRAM_PONG,
            (half *)(store_rgba_ping_nram + (store_h_repeat % 2) * nram_pong), store_h_remain,
            load_weight_pad);
        __sync_cluster();
        storeOutputFromSRAM2GDRAM(dst_addr + store_h_repeat * max_store_h * dst_stride,
                                  output_ping_sram + (store_h_repeat % 2) * SRAM_PONG, dst_stride,
                                  load_weight_pad, store_data_size, store_h_remain);
      }
      __sync_cluster();
    }
    cur_y_addr += compute_w;
    cur_uv_addr += compute_w;
    dst_addr += store_rgba;
    // current mask addr fix
    mask_left_addr += load_mask;
    mask_right_addr += load_mask;
    // current weight addr fix
    weight_right_addr += load_weight;
    // dst_idx_h fix
    dst_idx_h_start = h_cluster_start;
    dst_idx_h_end   = 0;
  }
}

__mlu_func__ void strategyOfBatchPartitionCluster(const int32_t batch_size,
                                                  const bool is_batch_split,
                                                  int32_t &batch_start,
                                                  int32_t &batch_end) {
  if (is_batch_split) {
    int32_t batch_seg = batch_size / taskDimY;
    int32_t batch_rem = batch_size % taskDimY;
    batch_start       = taskIdY * batch_seg + (taskIdY < batch_rem ? taskIdY : batch_rem);
    batch_end         = batch_start + batch_seg + (taskIdY < batch_rem ? 1 : 0);
  } else {
    batch_start = 0;
    batch_end   = batch_size;
  }
}

__mlu_func__ void strategyOfHeightPartitionCluster(const int32_t height,
                                                   const bool is_batch_split,
                                                   int32_t &h_cluster_num,
                                                   int32_t &h_cluster_start) {
  if (is_batch_split) {
    h_cluster_num   = height;
    h_cluster_start = 0;
  } else {
    int32_t height_seg = height / taskDimY;
    int32_t height_rem = height % taskDimY;
    h_cluster_num      = height_seg + (taskIdY < height_rem ? 1 : 0);
    h_cluster_start    = taskIdY * height_seg + (taskIdY < height_rem ? taskIdY : height_rem);
  }
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
 *  @param[in]  dst_channel
 *    Input. channel number of destination pixel format.
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
 *  @param[in]  is_batch_split
 *    Input. split policy.
 */
__mlu_global__ void MLUUnion1KernelResizeConvert300(void **src_y_gdram,
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
                                                    void **copy_filter_gdram,
                                                    bool is_batch_split) {
  /**---------------------- Initialization ----------------------**/
  __mlu_shared__ uint8_t sram_buffer[SRAM_SIZE];
  __nram__ int8_t nram_buffer[NRAM_SIZE];

  __wram__ int16_t yuv_filter_wram[kNumFilterChOut * kNumFilterHeightIn * kNumFilterChIn];
  __wram__ int8_t copy_filter_wram[kNumMultLimit * kNumLt * kNumLt];

  // Memory usage
  // Put all const variables(bias, weight, mask) at front of the buffer
  // so that space after them can be used freely without concerning
  // overwritting const variables
  half *yuv_bias_nram = (half *)nram_buffer;

  // fill background color
  uint8_t *round_nram       = (uint8_t *)(yuv_bias_nram + kNumFilterChOut);
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
      round_nram[dst_channel * i + cidx] = ((unsigned char *)fill_color_gdram)[cidx];
    }
  }

  uint8_t *pad_nram       = round_nram + fill_color_align;
  half *compute_temp_nram = (half *)pad_nram;

  // load yuv2rgba filter and bias
  half *yuv_filter_sram = (half *)sram_buffer;
  half *yuv_bias_sram   = yuv_filter_sram + (kNumFilterChOut * kNumFilterHeightIn * kNumFilterChIn);
  loadCvtFilter((half *)yuv_filter_wram, (half *)yuv_bias_nram, (half *)yuv_filter_sram,
                (half *)yuv_bias_sram, (half *)convert_filter_gdram, (half *)convert_bias_gdram);

  // height segment
  int32_t h_cluster_start = 0;
  int32_t h_cluster_num   = 0;
  // batch segment
  int32_t start_batch = 0;
  int32_t end_batch   = 0;

  // batch segment policy
  strategyOfBatchPartitionCluster(batch_size, is_batch_split, start_batch, end_batch);

  // for batch_size
  for (int32_t batch = start_batch; batch < end_batch; ++batch) {
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

    uint32_t dst_stride = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 6];
    uint32_t dst_height = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 7];
    uint32_t dst_roi_x  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 8];
    uint32_t dst_roi_y  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 9];
    uint32_t dst_roi_w  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 10];
    uint32_t dst_roi_h  = ((uint32_t *)shape_gdram)[batch * kNumShapeInfo + 11];

    if (coreId != 0x80) {
      fillBackgroundColor(dst_addr, pad_nram, round_nram, is_batch_split, fill_color_align,
                          dst_stride / dst_channel, dst_height, dst_channel, dst_stride, dst_roi_x,
                          dst_roi_y, dst_roi_w, dst_roi_h);
    }

    // height segment policy
    strategyOfHeightPartitionCluster(dst_roi_h, is_batch_split, h_cluster_num, h_cluster_start);

    // src_addr and dst_addr fix
    dst_addr = dst_addr + (dst_roi_y + h_cluster_start) * dst_stride + dst_roi_x * dst_channel;

    resizeConvert5PipelineMode(dst_addr, src_y_addr, src_uv_addr, copy_filter_addr, mask_left_addr,
                               mask_right_addr, weight_right_addr, sram_buffer, yuv_bias_nram,
                               compute_temp_nram, yuv_filter_wram, copy_filter_wram, src_stride,
                               src_roi_x, src_roi_y, src_roi_w, src_roi_h, dst_stride, dst_roi_w,
                               dst_roi_h, dst_channel, h_cluster_num, h_cluster_start);
  }
}
#else
__mlu_global__ void MLUUnion1KernelResizeConvert300(void **src_y_gdram,
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
                                                    void **copy_filter_gdram,
                                                    bool is_batch_split) {}
#endif

#endif  // PLUGIN_RESIZE_YUV_TO_RGBA_MLU300_KERNEL_H_
