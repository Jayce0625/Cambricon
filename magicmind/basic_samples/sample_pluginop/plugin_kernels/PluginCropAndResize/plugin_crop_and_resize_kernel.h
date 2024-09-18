/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_CROP_AND_RESIZE_KERNEL_H_
#define PLUGIN_CROP_AND_RESIZE_KERNEL_H_
#include <stdio.h>
#include "cnrt.h"

void CropAndResizeEnqueue(cnrtQueue_t queue,
                          void *dst_gdram,
                          void *src_gdram,
                          void *cropParams_gdram,
                          void *roiNums,
                          void *padValues,
                          int s_col,
                          int s_row,
                          int d_col_final,
                          int d_row_final,
                          int input2half,
                          int output2uint,
                          int batchNum,
                          int keepAspectRatio);
#endif  // PLUGIN_CROP_AND_RESIZE_KERNEL_H_
