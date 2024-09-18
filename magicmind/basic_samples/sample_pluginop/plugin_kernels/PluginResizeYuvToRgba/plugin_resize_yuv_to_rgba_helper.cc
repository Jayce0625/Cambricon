/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <cmath>
#include "third_party/half/half.h"
#include "plugin_resize_yuv_to_rgba_macro.h"
#include "plugin_resize_yuv_to_rgba_helper.h"

using half_float::half;

magicmind::Status getPixFmtChannelNum(magicmind::PixelFormat pixfmt, int32_t *chn_num) {
  switch (pixfmt) {
    case magicmind::PIX_FMT_GRAY:
    case magicmind::PIX_FMT_NV12:
    case magicmind::PIX_FMT_NV21:
      *chn_num = 1;
      break;
    case magicmind::PIX_FMT_RGB:
    case magicmind::PIX_FMT_BGR:
      *chn_num = 3;
      break;
    case magicmind::PIX_FMT_RGBA:
    case magicmind::PIX_FMT_BGRA:
    case magicmind::PIX_FMT_ARGB:
    case magicmind::PIX_FMT_ABGR:
      *chn_num = 4;
      break;
    default: {
      std::string temp =
          "[PluginResizeYuvToRgba] Got unknown PixelFormat. A valid PixelFormat must belong to "
          "[0, 8], but now got " +
          std::to_string((int32_t)pixfmt);
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
  }
  return magicmind::Status::OK();
}

magicmind::Status prepareConvertData(magicmind::ColorSpace color_space,
                                     magicmind::PixelFormat src_pixfmt,
                                     magicmind::PixelFormat dst_pixfmt,
                                     void *convert_data) {
  int U_idx, V_idx;

  switch (src_pixfmt) {
    case magicmind::PIX_FMT_NV12: {
      U_idx = 0;
      V_idx = 1;
      break;
    }
    case magicmind::PIX_FMT_NV21: {
      U_idx = 1;
      V_idx = 0;
      break;
    }
    default: {
      std::string temp =
          "[PluginResizeYuvToRgba] Input PixelFormat must be PIX_FMT_NV12(1) or "
          "PIX_FMT_NV21(2), but now got" +
          std::to_string((int32_t)src_pixfmt);
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
  }

  int16_t R_Y_weight, R_U_weight, R_V_weight, R_offset;
  int16_t G_Y_weight, G_U_weight, G_V_weight, G_offset;
  int16_t B_Y_weight, B_U_weight, B_V_weight, B_offset;
  int16_t A_offset = 0x5BF8;

  switch (color_space) {
    case magicmind::COLOR_SPACE_BT_601: {
      R_Y_weight = 0x253F;
      R_U_weight = 0x0000;
      R_V_weight = 0x3312;
      R_offset = 0xDAF7;
      G_Y_weight = 0x253F;
      G_U_weight = 0xF37D;
      G_V_weight = 0xE5FC;
      G_offset = 0x583C;
      B_Y_weight = 0x253F;
      B_U_weight = 0x4093;
      B_V_weight = 0x0000;
      B_offset = 0xDC54;
      break;
    }
    case magicmind::COLOR_SPACE_BT_709: {
      R_Y_weight = 0x253F;
      R_U_weight = 0x0000;
      R_V_weight = 0x3960;
      R_offset = 0xDBC1;
      G_Y_weight = 0x253F;
      G_U_weight = 0xF92F;
      G_V_weight = 0xEEE9;
      G_offset = 0x54D0;
      B_Y_weight = 0x253F;
      B_U_weight = 0x43AE;
      B_V_weight = 0x0000;
      B_offset = 0xDC85;
      break;
    }
    default: {
      std::string temp =
          "[PluginResizeYuvToRgba] Got unknown ColorSpace. A valid PixelFormat must belong to "
          "[0, 1], but now got " +
          std::to_string((int32_t)color_space);
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
  }

  int R_idx, G_idx, B_idx, A_idx;

  switch (dst_pixfmt) {
    case magicmind::PIX_FMT_RGB:
    case magicmind::PIX_FMT_RGBA: {
      R_idx = 0;
      G_idx = 1;
      B_idx = 2;
      A_idx = 3;
      break;
    }
    case magicmind::PIX_FMT_BGR:
    case magicmind::PIX_FMT_BGRA: {
      R_idx = 2;
      G_idx = 1;
      B_idx = 0;
      A_idx = 3;
      break;
    }
    case magicmind::PIX_FMT_ARGB: {
      R_idx = 1;
      G_idx = 2;
      B_idx = 3;
      A_idx = 0;
      break;
    }
    case magicmind::PIX_FMT_ABGR: {
      R_idx = 3;
      G_idx = 2;
      B_idx = 1;
      A_idx = 0;
      break;
    }
    default: {
      std::string temp =
          "[PluginResizeYuvToRgba] Output PixelFormat must be PIX_FMT_RGB(3), PIX_FMT_BGR(4), "
          "PIX_FMT_RGBA(5), PIX_FMT_BGRA(6), PIX_FMT_ARGB(7), or PIX_FMT_ABGR(8), but now got" +
          std::to_string((int32_t)dst_pixfmt);
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
  }

  // YUV SP420 TO RGBX USE CONV INST
  // FILTER SHAPE IS kNumFilterChOut * 2 * 1 * kNumFilterChIn (kNumFilterChOut * KH * KW *
  // kNumFilterChIn)
  int kernel_len = 2 * kNumFilterChIn;
  int kernel_co = kNumFilterChOut;
  int kernel_size = kNumFilterChOut * 2 * kNumFilterChIn;
  for (int lt = 0; lt < kNumLt; lt++) {
    for (int idx = 0; idx < (kernel_co / kNumLt); idx++) {
      // (lt + idx * kNumLt) present logic co
      // (lt * kernel_co / kNumLt + idx) present co in ddr
      int logic_co = lt + idx * kNumLt;
      int ddr_co = lt * (kernel_co / kNumLt) + idx;

      int offsetY = ddr_co * kernel_len + logic_co / 4;
      int offsetU = offsetY + kNumFilterChIn - (logic_co / 4) % 2 + U_idx;
      int offsetV = offsetY + kNumFilterChIn - (logic_co / 4) % 2 + V_idx;

      // Y
      ((int16_t *)convert_data)[offsetY] = ((logic_co % 4) == R_idx) * R_Y_weight +
                                           ((logic_co % 4) == G_idx) * G_Y_weight +
                                           ((logic_co % 4) == B_idx) * B_Y_weight;

      // U
      ((int16_t *)convert_data)[offsetU] = ((logic_co % 4) == R_idx) * R_U_weight +
                                           ((logic_co % 4) == G_idx) * G_U_weight +
                                           ((logic_co % 4) == B_idx) * B_U_weight;

      // V
      ((int16_t *)convert_data)[offsetV] = ((logic_co % 4) == R_idx) * R_V_weight +
                                           ((logic_co % 4) == G_idx) * G_V_weight +
                                           ((logic_co % 4) == B_idx) * B_V_weight;

      // bias
      ((int16_t *)convert_data)[kernel_size + logic_co] =
          ((logic_co % 4) == R_idx) * R_offset + ((logic_co % 4) == G_idx) * G_offset +
          ((logic_co % 4) == B_idx) * B_offset + ((logic_co % 4) == A_idx) * A_offset;
    }
  }

  return magicmind::Status::OK();
}

magicmind::Status prepareShapeData(const int32_t *src_shapes,
                                   const int32_t *src_rois,
                                   const int32_t *dst_shapes,
                                   const int32_t *dst_rois,
                                   int32_t batch_size,
                                   int32_t src_channel,
                                   int32_t dst_channel,
                                   int32_t pad_method,
                                   void *&shape_data_mlu,
                                   void *&shape_data_cpu) {
  // pad_method:
  //  - 0: no padding, i.e., does not keep aspect ratio
  //  - 1: padding bottom/top or left/right, i.e., keep images at centor of output
  //  - 2: padding bottom or right, i.e., keep images at top-left corner of output

  // batch loop
  for (int32_t batch_iter = 0; batch_iter < batch_size; batch_iter++) {
    int32_t cur_src_stride = src_shapes[batch_iter * 2 + 0] * src_channel;
    int32_t cur_src_roi_x = src_rois[batch_iter * 4 + 0];
    int32_t cur_src_roi_y = src_rois[batch_iter * 4 + 1];
    int32_t cur_src_roi_w = src_rois[batch_iter * 4 + 2];
    int32_t cur_src_roi_h = src_rois[batch_iter * 4 + 3];

    int32_t cur_dst_stride = dst_shapes[batch_iter * 2 + 0] * dst_channel;
    int32_t cur_dst_roi_x = dst_rois[batch_iter * 4 + 0];
    int32_t cur_dst_roi_y = dst_rois[batch_iter * 4 + 1];
    int32_t cur_dst_roi_w = dst_rois[batch_iter * 4 + 2];
    int32_t cur_dst_roi_h = dst_rois[batch_iter * 4 + 3];

    int32_t new_dst_roi_x = cur_dst_roi_x;
    int32_t new_dst_roi_y = cur_dst_roi_y;
    int32_t new_dst_roi_w = cur_dst_roi_w;
    int32_t new_dst_roi_h = cur_dst_roi_h;
    if (pad_method > 0) {
      // padding bottom/top when
      float src_ar = (float)cur_src_roi_w / cur_src_roi_h;
      float dst_ar = (float)cur_dst_roi_w / cur_dst_roi_h;
      if (src_ar > dst_ar) {
        // dst_roi_h is relatively larger than src_roi_h, update dst_roi_h.
        new_dst_roi_h = (int32_t)std::round((float)cur_dst_roi_w / cur_src_roi_w * cur_src_roi_h);
      } else if (src_ar < dst_ar) {
        // dst_roi_w is relatively larger than src_roi_w, update dst_roi_h.
        new_dst_roi_w = (int32_t)std::round((float)cur_dst_roi_h / cur_src_roi_h * cur_src_roi_w);
      } else {
        // src_ar == dst_ar, no actions needed.
      }
    }

    if (pad_method == 1) {
      new_dst_roi_x += ((cur_dst_roi_w - new_dst_roi_w) / 2);
      new_dst_roi_y += ((cur_dst_roi_h - new_dst_roi_h) / 2);
    }

    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 0] = cur_src_stride;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 1] = src_shapes[batch_iter * 2 + 1];
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 2] = cur_src_roi_x;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 3] = cur_src_roi_y;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 4] = cur_src_roi_w;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 5] = cur_src_roi_h;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 6] = cur_dst_stride;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 7] = dst_shapes[batch_iter * 2 + 1];
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 8] = new_dst_roi_x;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 9] = new_dst_roi_y;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 10] = new_dst_roi_w;
    ((int32_t *)shape_data_cpu)[batch_iter * kNumShapeInfo + 11] = new_dst_roi_h;
  }

  // fix shape data ddr address
  size_t shape_size = CEIL_ALIGN_NUM(batch_size * kNumShapeInfo, int32_t);
  shape_data_cpu = (void *)(((int32_t *)shape_data_cpu) + shape_size);
  shape_data_mlu = (void *)(((int32_t *)shape_data_mlu) + shape_size);

  return magicmind::Status::OK();
}

magicmind::Status prepareInterpData(const int32_t *shape_data_cpu,
                                    const int32_t *src_rois,
                                    const int32_t *dst_rois,
                                    int32_t batch_size,
                                    magicmind::PixelFormat dst_pixfmt,
                                    void *&interp_data_mlu,
                                    void *&interp_data_cpu) {
  size_t channel = kNumDefaultChannel;
  if (dst_pixfmt == magicmind::PIX_FMT_GRAY) {
    channel = 1;
  }
  int32_t dst_channel = 0;
  magicmind::Status status = getPixFmtChannelNum(dst_pixfmt, &dst_channel);
  if (status != magicmind::Status::OK()) {
    return status;
  }

  // mask and weight 2nd pointer
  size_t interp_2nd_ptr_size = CEIL_ALIGN_NUM(3 * batch_size, half *);

  half **mask_2nd_ptr_cpu = (half **)(interp_data_cpu);
  half **weight_2nd_ptr_cpu = (half **)(mask_2nd_ptr_cpu + batch_size * 2);
  half *cur_interp_data_cpu = (half *)(((half **)interp_data_cpu) + interp_2nd_ptr_size);

  // half **mask_2nd_ptr_mlu = (half **)(interp_data_mlu);
  // half **weight_2nd_ptr_mlu = (half **)(mask_2nd_ptr_mlu + batch_size * 2);
  half *cur_interp_data_mlu = (half *)(((half **)interp_data_mlu) + interp_2nd_ptr_size);

  // malloc temp list to determine whether to reuse data
  int *mult_list = (int *)malloc(batch_size * sizeof(int));
  int32_t *src_roi_x_list = (int32_t *)malloc(batch_size * sizeof(int32_t));
  int32_t *src_roi_w_list = (int32_t *)malloc(batch_size * sizeof(int32_t));
  int32_t *dst_roi_w_list = (int32_t *)malloc(batch_size * sizeof(int32_t));

  // PARAM_CHECK(API_NAME, mult_list != NULL);
  // PARAM_CHECK(API_NAME, src_roi_x_list != NULL);
  // PARAM_CHECK(API_NAME, src_roi_w_list != NULL);
  // PARAM_CHECK(API_NAME, dst_roi_w_list != NULL);

  // batch loop
  for (int32_t batch_iter = 0; batch_iter < batch_size; batch_iter++) {
    // int32_t cur_src_roi_x = src_rois[batch_iter * 4 + 0];
    // int32_t cur_src_roi_w = src_rois[batch_iter * 4 + 2];
    int32_t cur_src_roi_x = shape_data_cpu[batch_iter * kNumShapeInfo + 2];
    int32_t cur_src_roi_w = shape_data_cpu[batch_iter * kNumShapeInfo + 4];

    int32_t cur_src_roi_w_ = (cur_src_roi_x % 2 + cur_src_roi_w + 1) / 2 * 2;

    // int32_t cur_dst_roi_w = dst_rois[batch_iter * 4 + 2];
    int32_t cur_dst_roi_w = shape_data_cpu[batch_iter * kNumShapeInfo + 10];

    // if src_roi_w < cur_dst_roi_w, may need to copy src pixel
    // e.g. mult = 2 r1g1b1a1 r2g2b2a2 -> r1g1b1a1 r1g1b1a1 r2g2b2a2 r2g2b2a2
    // if src_roi_w >= cur_dst_roi_w, don't need to copy src pixel
    int mult = (int)(cur_src_roi_w < cur_dst_roi_w) *
                   (ceil(1.5 * (float)cur_dst_roi_w / cur_src_roi_w + 0.5)) +
               (int)(cur_src_roi_w >= cur_dst_roi_w);

    // save information to determine whether to reuse data
    mult_list[batch_iter] = mult;
    src_roi_x_list[batch_iter] = cur_src_roi_x;
    src_roi_w_list[batch_iter] = cur_src_roi_w;
    dst_roi_w_list[batch_iter] = cur_dst_roi_w;

    // reuse computed mask and weight or compute current mask and weight
    bool repeated = false;

    // reuse precomputed mask and weight
    for (int32_t prev_batch = 0; prev_batch < batch_iter; prev_batch++) {
      if (((cur_src_roi_x % 2) == (src_roi_x_list[prev_batch] % 2)) &&
          (cur_src_roi_w == src_roi_w_list[prev_batch]) &&
          (cur_dst_roi_w == dst_roi_w_list[prev_batch])) {
        // reuse mask data
        mask_2nd_ptr_cpu[batch_iter * 2] = mask_2nd_ptr_cpu[prev_batch * 2];
        mask_2nd_ptr_cpu[batch_iter * 2 + 1] = mask_2nd_ptr_cpu[prev_batch * 2 + 1];

        // reuse weight data
        weight_2nd_ptr_cpu[batch_iter] = weight_2nd_ptr_cpu[prev_batch];

        repeated = true;
        break;
      }
    }

    // compute current mask and weight
    if (!repeated) {
      size_t cur_mask_size = CEIL_ALIGN_NUM(mult * cur_src_roi_w_ * channel, half);
      size_t cur_weight_size = CEIL_ALIGN_NUM(cur_dst_roi_w * dst_channel, half);

      // cpu pointer offset
      half *cur_mask_left_ptr_cpu = (half *)(cur_interp_data_cpu);
      half *cur_mask_right_ptr_cpu = (half *)(cur_mask_left_ptr_cpu + cur_mask_size);
      half *cur_weight_right_ptr_cpu = (half *)(cur_mask_right_ptr_cpu + cur_mask_size);
      cur_interp_data_cpu = (half *)(cur_weight_right_ptr_cpu + cur_weight_size);

      // mlu pointer offset
      half *cur_mask_left_ptr_mlu = (half *)(cur_interp_data_mlu);
      half *cur_mask_right_ptr_mlu = (half *)(cur_mask_left_ptr_mlu + cur_mask_size);
      half *cur_weight_right_ptr_mlu = (half *)(cur_mask_right_ptr_mlu + cur_mask_size);
      cur_interp_data_mlu = (half *)(cur_weight_right_ptr_mlu + cur_weight_size);

      // compute w scale
      float cur_scale_w = float(cur_src_roi_w) / cur_dst_roi_w;
      float src_w_base = 0.5 * cur_scale_w - 0.5;

      float src_w = src_w_base;
      int src_w_int = -1;
      int src_w_int_prev = -1;

      float right_weight = 0.0;

      int32_t mask_left_index = 0;
      int32_t mask_right_index = 0;

      for (int32_t dst_w_iter = 0; dst_w_iter < cur_dst_roi_w; dst_w_iter++) {
        // compute dst_w index corresponding src w index
        src_w = dst_w_iter * cur_scale_w + src_w_base;
        src_w = src_w < 0 ? 0 : src_w;
        src_w = src_w > (cur_src_roi_w - 1) ? cur_src_roi_w - 1 : src_w;
        src_w_int = floor(src_w);

        // compute mask data
        if (src_w_int == src_w_int_prev) {
          mask_left_index++;
        } else {
          mask_left_index = (src_w_int + cur_src_roi_x % 2) * mult;
        }
        mask_right_index = mask_left_index + mult;

        // set left mask
        for (int32_t chn_idx = 0; chn_idx < dst_channel; ++chn_idx) {
          cur_mask_left_ptr_cpu[mask_left_index * channel + chn_idx] = 1;
        }

        // set right mask
        if (mask_right_index < (cur_src_roi_w + cur_src_roi_x % 2) * mult) {
          for (int32_t chn_idx = 0; chn_idx < dst_channel; ++chn_idx) {
            cur_mask_right_ptr_cpu[mask_left_index * channel + chn_idx] = 1;
          }
        }

        // compute weight data
        right_weight = src_w - src_w_int;

        for (int32_t chn_idx = 0; chn_idx < dst_channel; ++chn_idx) {
          cur_weight_right_ptr_cpu[dst_w_iter * dst_channel + chn_idx] = right_weight;
        }

        // update data for next iter
        src_w_int_prev = src_w_int;
      }

      // set mlu pointer addr
      mask_2nd_ptr_cpu[batch_iter * 2] = cur_mask_left_ptr_mlu;
      mask_2nd_ptr_cpu[batch_iter * 2 + 1] = cur_mask_right_ptr_mlu;

      weight_2nd_ptr_cpu[batch_iter] = cur_weight_right_ptr_mlu;
    }
  }

  // free temp list
  if (mult_list != nullptr) {
    free(mult_list);
    mult_list = nullptr;
  }
  if (src_roi_x_list != nullptr) {
    free(src_roi_x_list);
    src_roi_x_list = nullptr;
  }
  if (src_roi_w_list != nullptr) {
    free(src_roi_w_list);
    src_roi_w_list = nullptr;
  }
  if (dst_roi_w_list != nullptr) {
    free(dst_roi_w_list);
    dst_roi_w_list = nullptr;
  }

  // update interp data addr
  interp_data_cpu = ((void *)cur_interp_data_cpu);
  interp_data_mlu = ((void *)cur_interp_data_mlu);

  return magicmind::Status::OK();
}

magicmind::Status prepareCopyData(const int32_t *shape_data_cpu,
                                  const int32_t *src_rois,
                                  const int32_t *dst_rois,
                                  const int32_t batch_size,
                                  magicmind::PixelFormat dst_pixfmt,
                                  void *&copy_data_mlu,
                                  void *&copy_data_cpu) {
  // dst pixel fomat
  size_t channel = kNumDefaultChannel;
  if (dst_pixfmt == magicmind::PIX_FMT_GRAY) {
    channel = 1;
  }

  // copy filter 2nd pointer
  size_t copy_filter_2nd_ptr_size = CEIL_ALIGN_NUM(batch_size, int8_t *);

  int8_t **copy_filter_2nd_ptr_cpu = (int8_t **)(copy_data_cpu);
  int8_t *cur_copy_data_cpu = (int8_t *)(copy_filter_2nd_ptr_cpu + copy_filter_2nd_ptr_size);

  int8_t **copy_filter_2nd_ptr_mlu = (int8_t **)(copy_data_mlu);
  int8_t *cur_copy_data_mlu = (int8_t *)(copy_filter_2nd_ptr_mlu + copy_filter_2nd_ptr_size);

  // malloc temp list to determine whether to reuse data
  int *mult_list = (int *)malloc(batch_size * sizeof(int));

  // PARAM_CHECK(API_NAME, mult_list != NULL);

  // batch loop
  for (int32_t batch_iter = 0; batch_iter < batch_size; batch_iter++) {
    // int32_t cur_src_roi_w = src_rois[batch_iter * 4 + 2];
    // int32_t cur_dst_roi_w = dst_rois[batch_iter * 4 + 2];
    int32_t cur_src_roi_w = shape_data_cpu[batch_iter * kNumShapeInfo + 4];
    int32_t cur_dst_roi_w = shape_data_cpu[batch_iter * kNumShapeInfo + 10];

    // if src_roi_w < cur_dst_roi_w, may need to copy src pixel
    // e.g. mult = 2 r1g1b1a1 r2g2b2a2 -> r1g1b1a1 r1g1b1a1 r2g2b2a2 r2g2b2a2
    // if src_roi_w >= cur_dst_roi_w, don't need to copy src pixel
    int mult =
        (int)(cur_src_roi_w < cur_dst_roi_w) * (ceil(1.5 * cur_dst_roi_w / cur_src_roi_w + 0.5)) +
        (int)(cur_src_roi_w >= cur_dst_roi_w);

    // save information to determine whether to reuse data
    mult_list[batch_iter] = mult;

    // reuse computed copy filter or compute current copy filter
    if ((mult > 1) && (mult <= kNumMultLimit)) {
      bool repeated = false;

      // reuse precomputed copy filter
      for (int32_t prev_batch = 0; prev_batch < batch_iter; prev_batch++) {
        if (mult == mult_list[prev_batch]) {
          copy_filter_2nd_ptr_cpu[batch_iter] = copy_filter_2nd_ptr_cpu[prev_batch];

          repeated = true;
          break;
        }
      }

      // compute current copy filter
      if (!repeated) {
        // cpu pointer offset
        size_t cur_copy_data_size = CEIL_ALIGN_NUM(kNumLt * mult * kNumLt, int8_t);

        int8_t *cur_copy_filter_cpu = (int8_t *)(cur_copy_data_cpu);
        cur_copy_data_cpu = (int8_t *)(cur_copy_filter_cpu + cur_copy_data_size);

        // mlu pointer offset
        int8_t *cur_copy_filter_mlu = (int8_t *)(cur_copy_data_mlu);
        cur_copy_data_mlu = (int8_t *)(cur_copy_filter_mlu + cur_copy_data_size);

        // lt data
        // USE CONV INST COPY src pixel
        // e.g. mult = 2 r1g1b1a1 r2g2b2a2 -> r1g1b1a1 r1g1b1a1 r2g2b2a2 r2g2b2a2
        // FILTER SHAPE IS (mult * kNumLt) * 1 * 1 * kNumLt (kNumFilterChOut * KH * KW *
        // kNumFilterChIn)
        int kernel_len = kNumLt;
        int kernel_co = mult * kNumLt;
        for (int lt = 0; lt < kNumLt; lt++) {
          for (int idx = 0; idx < (kernel_co / kNumLt); idx++) {
            // (lt + idx * kNumLt) present logic co
            // (lt * kernel_co / kNumLt + idx) present co in ddr
            int logic_co = lt + idx * kNumLt;
            int ddr_co = lt * (kernel_co / kNumLt) + idx;

            int data_offset = logic_co / (mult * channel) * channel + logic_co % channel;

            cur_copy_filter_cpu[ddr_co * kernel_len + data_offset] = 1;
          }
        }

        copy_filter_2nd_ptr_cpu[batch_iter] = cur_copy_filter_mlu;
      }
    }
  }

  // free temp list
  if (mult_list != nullptr) {
    free(mult_list);
    mult_list = nullptr;
  }

  // update copy data addr
  copy_data_cpu = ((void *)cur_copy_data_cpu);
  copy_data_mlu = ((void *)cur_copy_data_mlu);

  return magicmind::Status::OK();
}

magicmind::Status prepareWorkspace(const int32_t *src_shapes,
                                   const int32_t *src_rois,
                                   const int32_t *dst_shapes,
                                   const int32_t *dst_rois,
                                   int32_t batch_size,
                                   int32_t pad_method,
                                   magicmind::ColorSpace src_cspace,
                                   magicmind::ColorSpace dst_cspace,
                                   magicmind::PixelFormat src_pixfmt,
                                   magicmind::PixelFormat dst_pixfmt,
                                   void *workspace,
                                   void *workspace_cpu,
                                   void *&convert_filter_gdram,
                                   void *&convert_bias_gdram,
                                   void *&shape_gdram,
                                   void **&mask_gdram,
                                   void **&weight_gdram,
                                   void **&copy_filter_gdram) {
  magicmind::Status status;
  void *shape_data_mlu = workspace;
  void *shape_data_cpu = workspace_cpu;
  int32_t src_channel = 0;
  int32_t dst_channel = 0;
  status = getPixFmtChannelNum(src_pixfmt, &src_channel);
  if (status != magicmind::Status::OK()) {
    return status;
  }
  status = getPixFmtChannelNum(dst_pixfmt, &dst_channel);
  if (status != magicmind::Status::OK()) {
    return status;
  }

  // prepare convert data
  if (dst_pixfmt != magicmind::PIX_FMT_GRAY) {
    // update shape data ddr address
    size_t convert_size = CEIL_ALIGN_NUM(
        kNumFilterChOut * kNumFilterHeightIn * kNumFilterChIn + kNumFilterChOut, int16_t);
    shape_data_mlu = (void *)(((int16_t *)workspace) + convert_size);
    shape_data_cpu = (void *)(((int16_t *)workspace_cpu) + convert_size);

    // update convert mlu gdram address
    void *convert_data = workspace_cpu;
    convert_filter_gdram = (void *)workspace;
    convert_bias_gdram =
        (void *)(((int16_t *)workspace) + kNumFilterChOut * kNumFilterHeightIn * kNumFilterChIn);

    status = prepareConvertData(src_cspace, src_pixfmt, dst_pixfmt, convert_data);
    if (status != magicmind::Status::OK()) {
      return status;
    }
  }
  int32_t *shape_data_cpu_base_addr = (int32_t *)shape_data_cpu;

  // update shape mlu gdram address
  shape_gdram = (void *)shape_data_mlu;

  // prepare shape data and fix shape data ddr address
  status = prepareShapeData(src_shapes, src_rois, dst_shapes, dst_rois, batch_size, src_channel,
                            dst_channel, pad_method, shape_data_mlu, shape_data_cpu);

  if (status != magicmind::Status::OK()) {
    return status;
  }

  // update interp mlu gdram address
  void *interp_data_mlu = shape_data_mlu;
  void *interp_data_cpu = shape_data_cpu;
  mask_gdram = (void **)interp_data_mlu;
  weight_gdram = ((void **)interp_data_mlu) + batch_size * 2;

  // prepare interp mask and weight data and fix interp data ddr address
  status = prepareInterpData(shape_data_cpu_base_addr, src_rois, dst_rois, batch_size, dst_pixfmt,
                             interp_data_mlu, interp_data_cpu);
  if (status != magicmind::Status::OK()) {
    return status;
  }

  // prepare copy filter data for conv instruction
  // e.g. mult = 2 r1g1b1a1 r2g2b2a2 -> r1g1b1a1 r1g1b1a1 r2g2b2a2 r2g2b2a2
  void *copy_data_mlu = interp_data_mlu;
  void *copy_data_cpu = interp_data_cpu;
  copy_filter_gdram = (void **)copy_data_mlu;

  // prepare copy filter data and fix copy filter data ddr address
  status = prepareCopyData(shape_data_cpu_base_addr, src_rois, dst_rois, batch_size, dst_pixfmt,
                           copy_data_mlu, copy_data_cpu);

  if (status != magicmind::Status::OK()) {
    return status;
  }

  return magicmind::Status::OK();
}

magicmind::Status paramCheck(const int32_t *src_shapes,
                             const int32_t *src_rois,
                             const int32_t *dst_shapes,
                             const int32_t *dst_rois,
                             int32_t batch_size,
                             int32_t input_channel,
                             int32_t output_channel) {
  // check each batch
  for (int i = 0; i < batch_size; i++) {
    // src width and height must be postive even integer.
    int32_t src_w = src_shapes[2 * i + 0];
    int32_t src_h = src_shapes[2 * i + 1];
    if ((src_w % 2 != 0) || (src_w <= 0)) {
      std::string temp =
          "[PluginResizeYuvToRgba] input tensor W-dim must be positive even integer, but the " +
          std::to_string(i) + numberPostfix(i) + " input tensor's W-dim is " +
          std::to_string(src_w) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    if ((src_h % 2 != 0) || (src_h <= 0)) {
      std::string temp =
          "[PluginResizeYuvToRgba] input tensor H-dim must be positive even integer, but the " +
          std::to_string(i) + numberPostfix(i) + " input tensor's H-dim is " +
          std::to_string(src_h) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    // dst width and height must be postive integer.
    int32_t dst_w = dst_shapes[2 * i + 0];
    int32_t dst_h = dst_shapes[2 * i + 1];
    if (dst_w <= 0) {
      std::string temp =
          "[PluginResizeYuvToRgba] output tensor W-dim must be positive integer, but the " +
          std::to_string(i) + numberPostfix(i) + " output tensor's W-dim is " +
          std::to_string(dst_w) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    if (dst_h <= 0) {
      std::string temp =
          "[PluginResizeYuvToRgba] output tensor H-dim must be positive integer, but the " +
          std::to_string(i) + numberPostfix(i) + " output tensor's H-dim is " +
          std::to_string(dst_h) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    // roi must not exceed images boundaries
    int32_t src_roi_x = src_rois[4 * i + 0];
    int32_t src_roi_y = src_rois[4 * i + 1];
    int32_t src_roi_w = src_rois[4 * i + 2];
    int32_t src_roi_h = src_rois[4 * i + 3];
    if (src_roi_x < 0) {
      std::string temp = "[PluginResizeYuvToRgba] input_rois' roi_x must be non-negtive, but the " +
                         std::to_string(i) + numberPostfix(i) +
                         " input tensor's roi_x, i.e., input_rois[" + std::to_string(4 * i + 0) +
                         "] is " + std::to_string(src_roi_x) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    if (src_roi_y < 0) {
      std::string temp = "[PluginResizeYuvToRgba] input_rois' roi_y must be non-negtive, but the " +
                         std::to_string(i) + numberPostfix(i) +
                         " input tensor's roi_y, i.e., input_rois[" + std::to_string(4 * i + 1) +
                         "] is " + std::to_string(src_roi_y) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    if (src_roi_w <= 0) {
      std::string temp = "[PluginResizeYuvToRgba] input_rois' roi_w must be positive, but the " +
                         std::to_string(i) + numberPostfix(i) +
                         " input tensor's roi_w, i.e., input_rois[" + std::to_string(4 * i + 2) +
                         "] is " + std::to_string(src_roi_w) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    if (src_roi_h <= 0) {
      std::string temp = "[PluginResizeYuvToRgba] input_rois' roi_h must be positive, but the " +
                         std::to_string(i) + numberPostfix(i) +
                         " input tensor's roi_h, i.e., input_rois[" + std::to_string(4 * i + 3) +
                         "] is " + std::to_string(src_roi_h) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    if (src_roi_x + src_roi_w > src_w) {
      std::string temp =
          "[PluginResizeYuvToRgba] input_rois' roi_x + roi_w must be no larger than "
          "input_tensor's W-dim, but the " +
          std::to_string(i) + numberPostfix(i) +
          " input tensor's roi_x and roi_w, i.e., input_rois[" + std::to_string(4 * i + 0) +
          "] and input_rois[" + std::to_string(4 * i + 2) + "] are " + std::to_string(src_roi_x) +
          " and " + std::to_string(src_roi_w) + ", while input_tensor's W-dim is " +
          std::to_string(src_w) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    if (src_roi_y + src_roi_h > src_h) {
      std::string temp =
          "[PluginResizeYuvToRgba] input_rois' roi_y + roi_h must be no larger than "
          "input_tensor's H-dim, but the " +
          std::to_string(i) + numberPostfix(i) +
          " input tensor's roi_y and roi_h, i.e., input_rois[" + std::to_string(4 * i + 1) +
          "] and input_rois[" + std::to_string(4 * i + 3) + "] are " + std::to_string(src_roi_y) +
          " and " + std::to_string(src_roi_h) + ", while input_tensor's H-dim is " +
          std::to_string(src_h) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    int32_t dst_roi_x = dst_rois[4 * i + 0];
    int32_t dst_roi_y = dst_rois[4 * i + 1];
    int32_t dst_roi_w = dst_rois[4 * i + 2];
    int32_t dst_roi_h = dst_rois[4 * i + 3];
    if (dst_roi_x < 0) {
      std::string temp =
          "[PluginResizeYuvToRgba] output_rois' roi_x must be non-negtive, but the " +
          std::to_string(i) + numberPostfix(i) + " output tensor's roi_x, i.e., output_rois[" +
          std::to_string(4 * i + 0) + "] is " + std::to_string(dst_roi_x) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    if (dst_roi_y < 0) {
      std::string temp =
          "[PluginResizeYuvToRgba] output_rois' roi_y must be non-negtive, but the " +
          std::to_string(i) + numberPostfix(i) + " output tensor's roi_y, i.e., output_rois[" +
          std::to_string(4 * i + 1) + "] is " + std::to_string(dst_roi_y) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    if (dst_roi_w <= 0) {
      std::string temp = "[PluginResizeYuvToRgba] output_rois' roi_w must be positive, but the " +
                         std::to_string(i) + numberPostfix(i) +
                         " output tensor's roi_w, i.e., output_rois[" + std::to_string(4 * i + 2) +
                         "] is " + std::to_string(dst_roi_w) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }
    if (dst_roi_h <= 0) {
      std::string temp = "[PluginResizeYuvToRgba] output_rois' roi_h must be positive, but the " +
                         std::to_string(i) + numberPostfix(i) +
                         " output tensor's roi_h, i.e., output_rois[" + std::to_string(4 * i + 3) +
                         "] is " + std::to_string(dst_roi_h) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    if (dst_roi_x + dst_roi_w > dst_w) {
      std::string temp =
          "[PluginResizeYuvToRgba] output_rois' roi_x + roi_w must be no larger than "
          "output_tensor's W-dim, but the " +
          std::to_string(i) + numberPostfix(i) +
          " output tensor's roi_x and roi_w, i.e., output_rois[" + std::to_string(4 * i + 0) +
          "] and output_rois[" + std::to_string(4 * i + 2) + "] are " + std::to_string(dst_roi_x) +
          " and " + std::to_string(dst_roi_w) + ", while output_tensor's W-dim is " +
          std::to_string(dst_w) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    if (dst_roi_y + dst_roi_h > dst_h) {
      std::string temp =
          "[PluginResizeYuvToRgba] output_rois' roi_y + roi_h must be no larger than "
          "output_tensor's H-dim, but the " +
          std::to_string(i) + numberPostfix(i) +
          " output tensor's roi_y and roi_h, i.e., output_rois[" + std::to_string(4 * i + 1) +
          "] and output_rois[" + std::to_string(4 * i + 3) + "] are " + std::to_string(dst_roi_y) +
          " and " + std::to_string(dst_roi_h) + ", while output_tensor's H-dim is " +
          std::to_string(dst_h) + ".";
      magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
      return status;
    }

    // check expand and shrink mult
    if (dst_roi_w >= src_roi_w) {
      if (dst_roi_w > src_roi_w * kNumWidthExpandLimit) {
        std::string temp =
            "[PluginResizeYuvToRgba] The ratio of output roi_w over input roi_w must be no "
            "larger than " +
            std::to_string(kNumWidthExpandLimit) + ", but the ratio of " + std::to_string(i) +
            numberPostfix(i) + " output/input tensor, i.e., output_rois[" +
            std::to_string(4 * i + 2) + "] / input_rois[" + std::to_string(4 * i + 2) + "], is " +
            std::to_string(dst_roi_w) + " / " + std::to_string(src_roi_w) + ".";
        magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
        return status;
      }
    } else {
      if (src_roi_w > dst_roi_w * kNumWidthShrinkLimit) {
        std::string temp =
            "[PluginResizeYuvToRgba] The ratio of input roi_w over output roi_w must be no "
            "larger than " +
            std::to_string(kNumWidthShrinkLimit) + ", but the ratio of " + std::to_string(i) +
            numberPostfix(i) + " input/output tensor, i.e., input_rois[" +
            std::to_string(4 * i + 2) + "] / output_rois[" + std::to_string(4 * i + 2) + "], is " +
            std::to_string(src_roi_w) + " / " + std::to_string(dst_roi_w) + ".";
        magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
        return status;
      }
    }

    if (dst_roi_h >= src_roi_h) {
      if (dst_roi_h > src_roi_h * kNumHeightExpandLimit) {
        std::string temp =
            "[PluginResizeYuvToRgba] The ratio of output roi_h over input roi_h must be no "
            "larger than " +
            std::to_string(kNumHeightExpandLimit) + ", but the ratio of " + std::to_string(i) +
            numberPostfix(i) + " output/input tensor, i.e., output_rois[" +
            std::to_string(4 * i + 3) + "] / input_rois[" + std::to_string(4 * i + 3) + "], is " +
            std::to_string(dst_roi_h) + " / " + std::to_string(src_roi_h) + ".";
        magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
        return status;
      }
    } else {
      if (src_roi_h > dst_roi_h * kNumHeightShrinkLimit) {
        std::string temp =
            "[PluginResizeYuvToRgba] The ratio of input roi_h over output roi_h must be no "
            "larger than " +
            std::to_string(kNumHeightShrinkLimit) + ", but the ratio of " + std::to_string(i) +
            numberPostfix(i) + " input/output tensor, i.e., input_rois[" +
            std::to_string(4 * i + 3) + "] / output_rois[" + std::to_string(4 * i + 3) + "], is " +
            std::to_string(src_roi_h) + " / " + std::to_string(dst_roi_h) + ".";
        magicmind::Status status(magicmind::error::Code::INVALID_ARGUMENT, temp);
        return status;
      }
    }
  }

  return magicmind::Status::OK();
}

size_t getResizeConvertWorkspaceSize(const int32_t *src_shapes,
                                     const int32_t *src_rois,
                                     const int32_t *dst_shapes,
                                     const int32_t *dst_rois,
                                     int32_t batch_size,
                                     int32_t pad_method,
                                     int32_t dst_channel,
                                     magicmind::PixelFormat dst_pixfmt) {
  // get param
  size_t temp_size = 0;
  size_t channel = kNumDefaultChannel;

  if (dst_pixfmt == magicmind::PIX_FMT_GRAY) {
    channel = 1;
  }

  // convert filter and bias size
  if (dst_pixfmt != magicmind::PIX_FMT_GRAY) {
    temp_size += CEIL_ALIGN_SIZE(
        (kNumFilterHeightIn * kNumFilterChIn * kNumFilterChOut + kNumFilterChOut), int16_t);
  }

  // shape info size
  // (src stride, src_h, roi x, roi y, roi w, roi h)
  // (dst stride, dst_h, roi x, roi y, roi w, roi h)
  temp_size += CEIL_ALIGN_SIZE(batch_size * kNumShapeInfo, int32_t);

  // mask and weight 2nd pointer size
  temp_size += CEIL_ALIGN_SIZE(3 * batch_size, half *);

  // copy filter 2nd pointer size
  temp_size += CEIL_ALIGN_SIZE(batch_size, int8_t *);

  // malloc temp list to determine whether to reuse data
  int *mult_list = (int *)malloc(batch_size * sizeof(int));
  int32_t *src_roi_x_list = (int32_t *)malloc(batch_size * sizeof(int32_t));
  int32_t *src_roi_w_list = (int32_t *)malloc(batch_size * sizeof(int32_t));
  int32_t *dst_roi_w_list = (int32_t *)malloc(batch_size * sizeof(int32_t));

  // batch loop
  for (int32_t batch_iter = 0; batch_iter < batch_size; batch_iter++) {
    int32_t cur_src_roi_x = src_rois[batch_iter * 4 + 0];
    int32_t cur_src_roi_w = src_rois[batch_iter * 4 + 2];

    int32_t cur_src_roi_w_ = (cur_src_roi_x % 2 + cur_src_roi_w + 1) / 2 * 2;

    int32_t cur_dst_roi_w = dst_rois[batch_iter * 4 + 2];
    if (pad_method > 0) {  // when keep_aspect_ratio == true, dst_roi_w may be changed
      int32_t cur_dst_roi_h = dst_rois[batch_iter * 4 + 3];
      int32_t cur_src_roi_h = src_rois[batch_iter * 4 + 3];
      float src_ar = (float)cur_src_roi_h / cur_src_roi_w;
      float dst_ar = (float)cur_dst_roi_h / cur_dst_roi_w;
      if (src_ar > dst_ar) {  // cur_dst_roi_w is too large
        cur_dst_roi_w = std::round((float)cur_dst_roi_h * cur_src_roi_w / cur_src_roi_h);
      }
    }

    // if src_roi_w < cur_dst_roi_w, may need to copy src pixel
    // e.g. mult = 2 r1g1b1a1 r2g2b2a2 -> r1g1b1a1 r1g1b1a1 r2g2b2a2 r2g2b2a2
    // if src_roi_w >= cur_dst_roi_w, don't need to copy src pixel
    int mult = (int)(cur_src_roi_w < cur_dst_roi_w) *
                   (ceil(1.5 * (float)cur_dst_roi_w / cur_src_roi_w + 0.5)) +
               (int)(cur_src_roi_w >= cur_dst_roi_w);

    // save information to determine whether to reuse data
    mult_list[batch_iter] = mult;
    src_roi_x_list[batch_iter] = cur_src_roi_x;
    src_roi_w_list[batch_iter] = cur_src_roi_w;
    dst_roi_w_list[batch_iter] = cur_dst_roi_w;

    size_t mask_size = 2 * CEIL_ALIGN_SIZE(mult * cur_src_roi_w_ * channel, half);
    size_t weight_size = CEIL_ALIGN_SIZE(cur_dst_roi_w * dst_channel, half);

    for (int32_t prev_batch = 0; prev_batch < batch_iter; prev_batch++) {
      if (((cur_src_roi_x % 2) == (src_roi_x_list[prev_batch] % 2)) &&
          (cur_src_roi_w == src_roi_w_list[prev_batch]) &&
          (cur_dst_roi_w == dst_roi_w_list[prev_batch])) {
        mask_size = 0;
        weight_size = 0;
        break;
      }
    }

    size_t copy_filter_size = 0;
    if ((mult > 1) && (mult <= kNumMultLimit)) {
      copy_filter_size = CEIL_ALIGN_SIZE(mult * kNumLt * kNumLt, int8_t);

      for (int32_t prev_batch = 0; prev_batch < batch_iter; prev_batch++) {
        if (mult == mult_list[prev_batch]) {
          copy_filter_size = 0;
          break;
        }
      }
    }
    temp_size += mask_size + weight_size + copy_filter_size;
  }

  if (mult_list != nullptr) {
    free(mult_list);
    mult_list = nullptr;
  }
  if (src_roi_x_list != nullptr) {
    free(src_roi_x_list);
    src_roi_x_list = nullptr;
  }
  if (src_roi_w_list != nullptr) {
    free(src_roi_w_list);
    src_roi_w_list = nullptr;
  }
  if (dst_roi_w_list != nullptr) {
    free(dst_roi_w_list);
    dst_roi_w_list = nullptr;
  }

  return temp_size;
}
