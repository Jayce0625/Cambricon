/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_RESIZE_YUV_TO_RGBA_KERNEL_H_
#define PLUGIN_RESIZE_YUV_TO_RGBA_KERNEL_H_
#include <stdio.h>
#include "cnrt.h"

void ResizeYuvToRgbaEnqueue(cnrtQueue_t queue,
                            void **dst_gdram,
                            void **input_y,
                            void **input_uv,
                            void *shape_gdram,
                            void **mask_gdram,
                            void **weight_gdram,
                            void *fill_color_gdram,
                            void *yuv_filter_gdram,
                            void *yuv_bias_gdram,
                            void **expand_filter_gdram,
                            int32_t *dst_rois,
                            int batch_num,
                            int dst_channel);
#endif  // PLUGIN_RESIZE_YUV_TO_RGBA_KERNEL_H_
