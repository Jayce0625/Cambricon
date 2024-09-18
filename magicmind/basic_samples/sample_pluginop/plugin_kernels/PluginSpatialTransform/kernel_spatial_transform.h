/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_PLUGINOPS_PLUGIN_SPATIAL_TRANSFORM_OP_KERNEL_SPATIAL_TRANSFORM_H_
#define SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_PLUGINOPS_PLUGIN_SPATIAL_TRANSFORM_OP_KERNEL_SPATIAL_TRANSFORM_H_
#include <stdio.h>  // NOLINT
#include "cnrt.h"   // NOLINT

void SpatialTransformEnqueue(cnrtQueue_t queue,
                             void *dst_ddr,
                             void *src_ddr,
                             void *mat_ddr,
                             void *multable_value,
                             int batch_size,
                             int dst_h,
                             int dst_w,
                             int src_h,
                             int src_w,
                             int c,
                             int data_type,
                             int cal_type,
                             int mat_no_broadcast);
#endif  // SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_PLUGINOPS_PLUGIN_SPATIAL_TRANSFORM_OP_KERNEL_SPATIAL_TRANSFORM_H_
