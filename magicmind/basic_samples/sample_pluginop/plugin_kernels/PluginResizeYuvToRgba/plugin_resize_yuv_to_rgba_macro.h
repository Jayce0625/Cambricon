/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_RESIZE_YUV_TO_RGBA_MACRO_H_
#define PLUGIN_RESIZE_YUV_TO_RGBA_MACRO_H_

namespace magicmind {

typedef enum {
  PIX_FMT_GRAY = 0,
  PIX_FMT_NV12 = 1,
  PIX_FMT_NV21 = 2,
  PIX_FMT_RGB  = 3,
  PIX_FMT_BGR  = 4,
  PIX_FMT_RGBA = 5,
  PIX_FMT_BGRA = 6,
  PIX_FMT_ARGB = 7,
  PIX_FMT_ABGR = 8,
} PixelFormat;

typedef enum {
  COLOR_SPACE_BT_601 = 0,
  COLOR_SPACE_BT_709 = 1,
} ColorSpace;

}  // namespace magicmind

static const int kNumLt                = 64;
static const int kNumMultLimit         = 64;
static const int kNumDefaultChannel    = 4;
static const int kNumAlign             = 64;
static const int kNumFilterChIn        = 32;
static const int kNumFilterChOut       = 128;
static const int kNumFilterHeightIn    = 2;
static const int kNumShapeInfo         = 12;
static const int kNumWidthExpandLimit  = 600;
static const int kNumHeightExpandLimit = 8192;
static const int kNumWidthShrinkLimit  = 8192;
static const int kNumHeightShrinkLimit = 8192;

// Macros for mlu kernels
// ALIGN TO 1024 bit NUM
#define UINT8_PAD_SIZE 128
#define HALF_PAD_SIZE 64
#define RYTR_INT_ALIGN_UP(x, y) (((x) + (y)-1) / (y) * (y))
// RAM SIZE LIMIT
#define REM_FOR_STACK (128 * 1024)
#define THRESHOLD_SIZE_OF_UNION (64 * 1024)
#ifdef __BANG_ARCH__
#define MAX_NRAM_SIZE (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)
#define MAX_SRAM_SIZE (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)
#else
#define MAX_NRAM_SIZE (384 * 1024)
#define MAX_SRAM_SIZE (1920 * 1024)
#endif
#define SRAM_SIZE MAX_SRAM_SIZE
#define NRAM_SIZE MAX_NRAM_SIZE
#define SRAM_PONG (SRAM_SIZE / 2)

// Utils
#define BANG_MAX(x, y) ((x) > (y) ? (x) : (y))
#define BANG_MIN(x, y) ((x) > (y) ? (y) : (x))

// DDR DATA ALIGN TO 128 Byte for better DDR BandWidth
#define DDR_ALIGN_SIZE 128
#define CEIL_ALIGN(x, align) (((x) + (align)-1) / (align) * (align))
#define CEIL_ALIGN_SIZE(num, type) CEIL_ALIGN(((num) * sizeof(type)), DDR_ALIGN_SIZE)
#define CEIL_ALIGN_NUM(size, type) CEIL_ALIGN((size), DDR_ALIGN_SIZE / sizeof(type))

#endif  // PLUGIN_RESIZE_YUV_TO_RGBA_MACRO_H_
