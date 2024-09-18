/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include "common/data.h"
#include "common/macros.h"
#include "common/param.h"
#include "basic_samples/sample_pluginop/plugin_kernels/PluginResizeYuvToRgba/plugin_resize_yuv_to_rgba_macro.h"

typedef enum {
  RGBA_TO_RGBA = 0,      // Convert color space from RGBA to RGBA, not implemented.
  YUV_TO_RGBA_NV12 = 1,  // Convert color space from YUV420SP_NV12 to RGBA.
  YUV_TO_RGBA_NV21 = 2,  // Convert color space from YUV420SP_NV21 to RGBA.
  YUV_TO_BGRA_NV12 = 3,  // Convert color space from YUV420SP_NV12 to BGRA.
  YUV_TO_BGRA_NV21 = 4,  // Convert color space from YUV420SP_NV21 to BGRA.
  YUV_TO_ARGB_NV12 = 5,  // Convert color space from YUV420SP_NV12 to ARGB.
  YUV_TO_ARGB_NV21 = 6,  // Convert color space from YUV420SP_NV21 to ARGB.
  YUV_TO_ABGR_NV12 = 7,  // Convert color space from YUV420SP_NV12 to ABGR.
  YUV_TO_ABGR_NV21 = 8,  // Convert color space from YUV420SP_NV21 to ABGR.
  YUV_TO_RGB_NV12 = 9,   // Convert color space from YUV420SP_NV12 to RGB, not implemented.
  YUV_TO_RGB_NV21 = 10,  // Convert color space from YUV420SP_NV21 to RGB, not implemented.
  YUV_TO_BGR_NV12 = 11,  // Convert color space from YUV420SP_NV12 to BGR, not implemented.
  YUV_TO_BGR_NV21 = 12,  // Convert color space from YUV420SP_NV21 to BGR, not implemented.
  GRAY_TO_GRAY = 13      // Convert color space from GRAY to GRAY, not implemented.
} PluginColorCvt_t;

#define rand(x, y) (std::max((x), (std::rand() % (y))))
#define PRE_FIX(x, y) x##y
#define PIX_FMT_(x) PRE_FIX(PIX_FMT_, x)
#define CVT_CASE(IN, OUT)                                      \
  case (static_cast<uint32_t>(magicmind::PIX_FMT_(IN)) << 16 | \
        static_cast<uint32_t>(magicmind::PIX_FMT_(OUT)))

static PluginColorCvt_t pixFmtToColorCvt(magicmind::PixelFormat src, magicmind::PixelFormat dst) {
  uint32_t color_cvt_key = (static_cast<uint32_t>(src) << 16 | static_cast<uint32_t>(dst));
  switch (color_cvt_key) {
    CVT_CASE(RGBA, RGBA) : { return RGBA_TO_RGBA; };
    CVT_CASE(NV12, RGBA) : { return YUV_TO_RGBA_NV12; };  // 1 5
    CVT_CASE(NV21, RGBA) : { return YUV_TO_RGBA_NV21; };  // 2 5
    CVT_CASE(NV12, BGRA) : { return YUV_TO_BGRA_NV12; };  // 1 6
    CVT_CASE(NV21, BGRA) : { return YUV_TO_BGRA_NV21; };  // 2 6
    CVT_CASE(NV12, ARGB) : { return YUV_TO_ARGB_NV12; };  // 1 7
    CVT_CASE(NV21, ARGB) : { return YUV_TO_ARGB_NV21; };  // 2 7
    CVT_CASE(NV12, ABGR) : { return YUV_TO_ABGR_NV12; };  // 1 8
    CVT_CASE(NV21, ABGR) : { return YUV_TO_ABGR_NV21; };  // 2 8
    CVT_CASE(NV12, RGB) : { return YUV_TO_RGB_NV12; };
    CVT_CASE(NV21, RGB) : { return YUV_TO_RGB_NV21; };
    CVT_CASE(NV12, BGR) : { return YUV_TO_BGR_NV12; };
    CVT_CASE(NV21, BGR) : { return YUV_TO_BGR_NV21; };
    CVT_CASE(GRAY, GRAY) : { return GRAY_TO_GRAY; };
    default: { return RGBA_TO_RGBA; }
  }
}

// mapping the combination of roi_x/roi_w or roi_y/roi_h parity into a integer number.
//  - 0: even/even
//  - 1: even/odd
//  - 2: odd/even
//  - 3: odd/odd
static inline void mapRoICoords(int32_t *new_pos, int32_t *new_len, int32_t pos, int32_t len) {
  int32_t roi_case = (pos % 2) * 2 + (len % 2);
  switch (roi_case) {
    case 0:
      (*new_pos) = pos;
      (*new_len) = len;
      break;
    case 1:
      (*new_pos) = pos;
      (*new_len) = len + 1;
      break;
    case 2:
      (*new_pos) = pos - 1;
      (*new_len) = len + 2;
      break;
    case 3:
      (*new_pos) = pos - 1;
      (*new_len) = len + 1;
      break;
    default:
      // should never be triggered.
      (*new_pos) = pos;
      (*new_len) = len;
  }
}

// Using src_pos = (dst_pos + 0.5) * scale - 0.5
// where scale = src_h|w / dst_h|w.
// and pos is zero-based.
static inline void getSrcPosFromDst(int32_t *dst_pos_int,
                                    float *dst_pos_dec,
                                    int32_t src_pos,
                                    float scale,
                                    int32_t limit,
                                    int32_t offset) {
  float dst_pos = (src_pos + 0.5) * scale - 0.5;
  // 0 <= dst_pos <= limit, limit is usually (src_len - 1)
  dst_pos = std::max((float)0, std::min((float)limit, dst_pos));
  int32_t integer_part = (int)dst_pos + offset;
  float decimal_part = dst_pos - integer_part + offset;
  (*dst_pos_int) = integer_part;
  (*dst_pos_dec) = decimal_part;
}

static void ResizeYuvToRgbaCpuCompute(uint8_t *dst,
                                      const uint8_t *srcY,
                                      const uint8_t *srcUV,
                                      const int *roiRect,
                                      const uint8_t *fill_color,
                                      int s_col,
                                      int s_row,
                                      int d_row_final,
                                      int d_col_final,
                                      int d_chn,
                                      int batch_num,
                                      int padMethod,
                                      PluginColorCvt_t color) {
  for (int batchId = 0; batchId < batch_num; batchId++) {
    int roi_x = roiRect[batchId * 4 + 0];
    int roi_y = roiRect[batchId * 4 + 1];
    int roi_w = roiRect[batchId * 4 + 2];
    int roi_h = roiRect[batchId * 4 + 3];
    /*--------- rcop the src image----------*/

    // condition judgment
    int crop_x = 0;
    int crop_y = 0;
    int crop_w = 0;
    int crop_h = 0;

    mapRoICoords(&crop_x, &crop_w, roi_x, roi_w);
    mapRoICoords(&crop_y, &crop_h, roi_y, roi_h);
    int32_t crop_h_uv = crop_h / 2;
    // "g" for gray since y channel "=" gray
    unsigned char *crop_g = (unsigned char *)malloc(crop_w * crop_h * sizeof(char));
    unsigned char *crop_uv = (unsigned char *)malloc(crop_w * crop_h_uv * sizeof(char));

    // crop
    unsigned char *ptr_Y = crop_g;
    unsigned char *ptr_UV = crop_uv;
    // copy the Y data
    for (int i = 0; i < crop_h; i++) {
      int sy = i + crop_y;
      memcpy(ptr_Y + i * crop_w, srcY + sy * s_col + crop_x, crop_w * sizeof(char));
    }
    // copy the UV data
    for (int i = 0; i < crop_h_uv; i++) {
      int suv = i + crop_y / 2;
      memcpy(ptr_UV + i * crop_w, srcUV + suv * s_col + crop_x, crop_w * sizeof(char));
    }

    /*----------convert YUV to RGB-----------*/
    // R=Y×1.164+V×1.596-222.912
    // G=Y×1.164-U×0.392-V×0.813+135.616
    // B=Y×1.164+U×2.017-276.8
    int height = crop_h;
    int width = crop_w;

    float *yuv_rgb = (float *)malloc(height * width * 3 * sizeof(float));

    unsigned char Y = 0;
    unsigned char U = 0;
    unsigned char V = 0;
    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        Y = crop_g[width * r + c];
        if (color % 2) {
          U = crop_uv[(r / 2) * width + (c / 2) * 2];
          V = crop_uv[(r / 2) * width + (c / 2) * 2 + 1];
        } else {
          U = crop_uv[(r / 2) * width + (c / 2) * 2 + 1];
          V = crop_uv[(r / 2) * width + (c / 2) * 2];
        }
        for (int cc = 0; cc < 3; cc++) {
          if (color % 4 == 1 || color % 4 == 2) {
            switch (cc) {
              case 0: {  // R
                float r_value = 1.164 * Y + 1.596 * V - 222.912;
                if (r_value > 255)
                  r_value = 255;
                if (r_value < 0)
                  r_value = 0;
                yuv_rgb[r * width * 3 + c * 3 + cc] = r_value;
              } break;
              case 1: {  // G
                float g_value = 1.164 * Y - 0.392 * U - 0.813 * V + 135.616;
                if (g_value > 255)
                  g_value = 255;
                if (g_value < 0)
                  g_value = 0;
                yuv_rgb[r * width * 3 + c * 3 + cc] = g_value;
              } break;
              case 2: {  // B
                float b_value = 1.164 * Y + 2.017 * U - 276.8;
                if (b_value > 255)
                  b_value = 255;
                if (b_value < 0)
                  b_value = 0;
                yuv_rgb[r * width * 3 + c * 3 + cc] = b_value;
              } break;
            }
          }
          if (color % 4 == 0 || color % 4 == 3) {
            switch (cc) {
              case 2: {  // R
                float r_value = 1.164 * Y + 1.596 * V - 222.912;
                if (r_value > 255)
                  r_value = 255;
                if (r_value < 0)
                  r_value = 0;
                yuv_rgb[r * width * 3 + c * 3 + cc] = r_value;
              } break;
              case 1: {  // G
                float g_value = 1.164 * Y - 0.392 * U - 0.813 * V + 135.616;
                if (g_value > 255)
                  g_value = 255;
                if (g_value < 0)
                  g_value = 0;
                yuv_rgb[r * width * 3 + c * 3 + cc] = g_value;
              } break;
              case 0: {  // B
                float b_value = 1.164 * Y + 2.017 * U - 276.8;
                if (b_value > 255)
                  b_value = 255;
                if (b_value < 0)
                  b_value = 0;
                yuv_rgb[r * width * 3 + c * 3 + cc] = b_value;
              } break;
            }
          }
        }
      }
    }

    /*---------bilinear interpolation-----------*/

    int d_row = 0;
    int d_col = 0;
    int pad_mode = 0;
    int pad_half1 = 0;
    float src_aspect_ratio = 0;
    float dst_aspect_ratio = 0;
    if (padMethod > 0) {
      src_aspect_ratio = (float)roi_w / roi_h;
      dst_aspect_ratio = (float)d_col_final / d_row_final;
      if (src_aspect_ratio >= dst_aspect_ratio) {
        // pad top && bottom
        d_col = d_col_final;
        d_row = std::round((float)d_col_final / roi_w * roi_h);
        // src_aspect_ratio);
        pad_mode = 0;
        if (padMethod == 1) {
          pad_half1 = (d_row_final - d_row) / 2;
        } else if (padMethod == 2) {
          pad_half1 = 0;
        }
      } else {
        // pad left && right
        d_row = d_row_final;
        d_col = std::round((float)d_row_final / roi_h * roi_w);
        pad_mode = 1;
        if (padMethod == 1) {
          pad_half1 = (d_col_final - d_col) / 2;
        } else if (padMethod == 2) {
          pad_half1 = 0;
        }
      }
    } else {
      d_row = d_row_final;
      d_col = d_col_final;
    }

    uint8_t *tmp_dst = (uint8_t *)malloc(d_row * d_col * d_chn * sizeof(uint8_t));
    float scale_x = (float)roi_w / d_col;
    float scale_y = (float)roi_h / d_row;

    // Boundrary conditions:
    //  - pixels out of roi are not used(currently used)
    //  - pixels out of roi but inside source image can be used
    int32_t h_limit = roi_h - 1;  // idx is zero-based.
    int32_t w_limit = roi_w - 1;  // idx is zero-based.

    for (int i = 0; i < d_row; i++) {
      int32_t yi_0 = 0;
      float yd = 0;
      getSrcPosFromDst(&yi_0, &yd, i, scale_y, h_limit, (roi_y % 2));
      int32_t yi_1 = std::min(h_limit + (roi_y % 2), yi_0 + 1);

      for (int j = 0; j < d_col; j++) {
        int32_t xi_0 = 0;
        float xd = 0;
        getSrcPosFromDst(&xi_0, &xd, j, scale_x, w_limit, (roi_x % 2));
        int32_t xi_1 = std::min(w_limit + (roi_x % 2), xi_0 + 1);
        // interpolation for every channel
        for (int k = 0; k < 3; k++) {
          float index00 = yuv_rgb[yi_0 * width * 3 + xi_0 * 3 + k];
          float index01 = yuv_rgb[yi_0 * width * 3 + xi_1 * 3 + k];
          float index10 = yuv_rgb[yi_1 * width * 3 + xi_0 * 3 + k];
          float index11 = yuv_rgb[yi_1 * width * 3 + xi_1 * 3 + k];

          tmp_dst[i * d_col * d_chn + j * d_chn + k + (color > 4 && color < 9)] =
              std::round((float)((1 - xd) * (1 - yd) * index00 + xd * (1 - yd) * index01 +
                                 (1 - xd) * yd * index10 + xd * yd * index11));
          if (d_chn == 4) {
            tmp_dst[i * d_col * d_chn + j * d_chn + (3 - 3 * (color / 5))] = 255;
          }
        }
      }
    }

    // fill background
    for (int r = 0; r < d_row_final; r++) {
      for (int c = 0; c < d_col_final; c++) {
        for (int channel = 0; channel < d_chn; channel++) {
          dst[r * d_col_final * d_chn + c * d_chn + channel] = fill_color[channel];
        }
      }
    }
    // save
    if (padMethod > 0) {
      if (pad_mode == 0) {  // pad top && bottom
        memcpy(dst + d_col * pad_half1 * d_chn, tmp_dst, d_col * d_row * d_chn * sizeof(uint8_t));
      } else {  // pad left && right
        for (int i = 0; i < d_row; i++) {
          memcpy(dst + i * d_col_final * d_chn + pad_half1 * d_chn, tmp_dst + i * d_col * d_chn,
                 d_col * d_chn * sizeof(uint8_t));
        }
      }
    } else {
      memcpy(dst, tmp_dst, d_row_final * d_col_final * d_chn * sizeof(uint8_t));
    }

    // free source
    if (crop_g) {
      free(crop_g);
      crop_g = nullptr;
    }

    if (crop_uv) {
      free(crop_uv);
      crop_uv = nullptr;
    }

    if (yuv_rgb) {
      free(yuv_rgb);
      yuv_rgb = nullptr;
    }

    if (tmp_dst) {
      free(tmp_dst);
      tmp_dst = nullptr;
    }
  }  // for batch_num

  return;
}

class YUV2RGBHostArg : public ArgListBase {
  DECLARE_ARG(input_format, (int))
      ->SetDescription("Input format: 1:YUVV420SP_NV12, 2:YUV420SP_NV21")
      ->SetAlternative({"1", "2"});
  DECLARE_ARG(output_format, (int))
      ->SetDescription("Output format: 1:RGB, 2:BGR, 3:RGBA, 4:BGRA, 5:ARGB, 6:ABGR")
      ->SetAlternative({"1", "2", "3", "4", "5", "6"});
  DECLARE_ARG(uv_shapes, (std::vector<std::vector<int>>))->SetDescription("Input shapes for uv.");
  DECLARE_ARG(d_row, (int))->SetDescription("Output hight");
  DECLARE_ARG(d_col, (int))->SetDescription("Output weight");
  DECLARE_ARG(pad_method, (int))
      ->SetDescription(
          "To fill pad around or bottom-right corner, 0: not keep ratio, 1: keep ratio with valid "
          "input in the middle, 2: keep ratio with valid input in the left-top corner")
      ->SetAlternative({"0", "1", "2"});
  DECLARE_ARG(rois, (std::vector<int>))
      ->SetDescription("Rois {x,y,w,h}, x+w <= uv[w], y+h <= uv[h] * 2");
};

int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  YUV2RGBHostArg arg_reader;
  arg_reader.ReadIn(args);
  SLOG(INFO) << arg_reader.DebugString();
  auto uv_shapes = Value(arg_reader.uv_shapes());
  int total_batch_num = 0;
  std::vector<std::vector<int>> y_shapes(uv_shapes.size());
  auto rois = Value(arg_reader.rois());
  CHECK_EQ(rois.size(), 4);
  for (size_t i = 0; i < uv_shapes.size(); ++i) {
    CHECK_EQ(uv_shapes[i].size(), 4);
    total_batch_num += uv_shapes[i][0];
    y_shapes[i] = std::vector<int>({uv_shapes[i][0], uv_shapes[i][1] * 2, uv_shapes[i][2]});
    CHECK_LE((rois[1] + rois[3]), y_shapes[i][1]);
    CHECK_LE((rois[0] + rois[2]), y_shapes[i][2]);
  }
  auto input_format = Value(arg_reader.input_format());
  auto output_format = Value(arg_reader.output_format()) + 2;
  auto pad_method = Value(arg_reader.pad_method());
  int d_row = Value(arg_reader.d_row());
  int d_col = Value(arg_reader.d_col());

  PluginColorCvt_t color =
      pixFmtToColorCvt((magicmind::PixelFormat)input_format, (magicmind::PixelFormat)output_format);

  // fill_color: uint8_t type input
  std::vector<uint8_t> fill_color;
  if (output_format < 5) {  // rgb/bgr
    fill_color = std::vector<uint8_t>({1, 2, 3});
  } else if (output_format < 7) {  // rgba/bgra
    fill_color = std::vector<uint8_t>({1, 2, 3, 255});
  } else {
    fill_color = std::vector<uint8_t>({255, 1, 2, 3});
  }
  CHECK_VALID(
      WriteDataToFile("./fill_color", fill_color.data(), fill_color.size() * sizeof(uint8_t)));
  // y_data
  std::vector<std::vector<uint8_t>> y_data(y_shapes.size());
  for (size_t i = 0; i < y_data.size(); ++i) {
    y_data[i] = GenRand<uint8_t>(y_shapes[i][0] * y_shapes[i][1] * y_shapes[i][2], 0, 255, 0);
    CHECK_VALID(WriteDataToFile("./src_y" + std::to_string(i), y_data[i].data(),
                                y_data[i].size() * sizeof(uint8_t)));
  }

  // uv_data
  std::vector<std::vector<uint8_t>> uv_data(uv_shapes.size());
  for (size_t i = 0; i < uv_data.size(); ++i) {
    uv_data[i] = GenRand<uint8_t>(uv_shapes[i][0] * uv_shapes[i][1] * uv_shapes[i][2], 0, 255, 0);
    CHECK_VALID(WriteDataToFile("./src_uv" + std::to_string(i), uv_data[i].data(),
                                uv_data[i].size() * sizeof(uint8_t)));
  }
  // output_tensor_total_batch_num == total_batch_num
  std::vector<uint8_t> output_data(total_batch_num * d_row * d_col * fill_color.size() /*channel*/);
  // input_rois(Host)
  std::vector<int> roi_data;
  for (int i = 0; i < total_batch_num; ++i) {
    roi_data.insert(roi_data.end(), rois.begin(), rois.end());
  }
  CHECK_VALID(WriteDataToFile("./in_roi", roi_data.data(), roi_data.size() * sizeof(int)));
  int out_offset = d_col * d_row * fill_color.size();
  int cal_batch = 0;
  for (size_t tidx = 0; tidx < uv_shapes.size() /* input_num */; ++tidx) {
    int batch_num = uv_shapes[tidx][0];
    int y_offset = y_shapes[tidx][1] * y_shapes[tidx][2];
    int uv_offset = uv_shapes[tidx][1] * uv_shapes[tidx][2];
    // Compute for each single batch
    for (int batch = 0; batch < batch_num; ++batch) {
      ResizeYuvToRgbaCpuCompute(
          output_data.data() + cal_batch * out_offset, y_data[tidx].data() + y_offset * batch,
          uv_data[tidx].data() + uv_offset * batch, roi_data.data() + cal_batch * 4,
          fill_color.data(), y_shapes[tidx][2], y_shapes[tidx][1], d_row, d_col, fill_color.size(),
          1, pad_method, color);
      cal_batch++;
    }
  }
  CHECK_VALID(
      WriteDataToFile("./baseline", output_data.data(), output_data.size() * sizeof(uint8_t)));
  return 0;
}
