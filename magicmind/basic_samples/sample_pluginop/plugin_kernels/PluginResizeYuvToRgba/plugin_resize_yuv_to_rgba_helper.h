/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_RESIZE_YUV_TO_RGBA_HELPER_H_
#define PLUGIN_RESIZE_YUV_TO_RGBA_HELPER_H_
#include <map>
#include <string>
#include "mm_plugin.h"
#include "third_party/half/half.h"
#include "plugin_resize_yuv_to_rgba_macro.h"

static std::map<int32_t, std::string> kNumToEngMap{
    {0, "th"}, {1, "st"}, {2, "nd"}, {3, "rd"}, {4, "th"},
};

static inline std::string numberPostfix(int32_t idx) {
  auto postfix_iter = kNumToEngMap.find(idx);
  if (postfix_iter == kNumToEngMap.end()) {
    return "th";
  }
  return postfix_iter->second;
}

magicmind::Status getPixFmtChannelNum(magicmind::PixelFormat pixfmt, int32_t *chn_num);

magicmind::Status prepareConvertData(magicmind::ColorSpace color_space,
                                     magicmind::PixelFormat src_pixfmt,
                                     magicmind::PixelFormat dst_pixfmt,
                                     void *convert_data);

magicmind::Status prepareShapeData(const int32_t *src_shapes,
                                   const int32_t *src_rois,
                                   const int32_t *dst_shapes,
                                   const int32_t *dst_rois,
                                   int32_t batch_size,
                                   int32_t src_channel,
                                   int32_t dst_channel,
                                   void *&shape_data_mlu,
                                   void *&shape_data_cpu);

magicmind::Status prepareInterpData(const int32_t *shape_data_cpu,
                                    const int32_t *src_rois,
                                    const int32_t *dst_rois,
                                    int32_t batch_size,
                                    magicmind::PixelFormat dst_pixfmt,
                                    void *&interp_data_mlu,
                                    void *&interp_data_cpu);

magicmind::Status prepareCopyData(const int32_t *shape_data_cpu,
                                  const int32_t *src_rois,
                                  const int32_t *dst_rois,
                                  const int32_t batch_size,
                                  magicmind::PixelFormat dst_pixfmt,
                                  void *&copy_data_mlu,
                                  void *&copy_data_cpu);

magicmind::Status prepareWorkspace(const int32_t *src_shapes,
                                   const int32_t *src_rois,
                                   const int32_t *dst_shapes,
                                   const int32_t *dst_rois,
                                   int32_t batch_size,
                                   int32_t pad_mathod,
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
                                   void **&copy_filter_gdram);

magicmind::Status paramCheck(const int32_t *src_shapes,
                             const int32_t *src_rois,
                             const int32_t *dst_shapes,
                             const int32_t *dst_rois,
                             int32_t batch_size,
                             int32_t input_channel,
                             int32_t output_channel);

size_t getResizeConvertWorkspaceSize(const int32_t *src_shapes,
                                     const int32_t *src_rois,
                                     const int32_t *dst_shapes,
                                     const int32_t *dst_rois,
                                     int32_t batch_size,
                                     int32_t pad_method,
                                     int32_t dst_channel,
                                     magicmind::PixelFormat dst_fixfmt);
#endif  // SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_PLUGINOPS_PLUGIN_RESIZE_YUV_TO_RGBA_OP_PLUGIN_RESIZE_YUV_TO_RGBA_HELPER_H_
