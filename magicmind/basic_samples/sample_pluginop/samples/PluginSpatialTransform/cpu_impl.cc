/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "mm_common.h"
#include "mm_status.h"
#include "common/logger.h"
#include "common/macros.h"
#include "common/data.h"
#include "common/param.h"
#include "third_party/half/half.h"
static float SpatialTransformCpuForward(float *pic, float x, float y, int H, int W) {
  float res = (float)0.;
  int m, n;
  float w;

  // Top-left point
  m = std::floor(x);
  n = std::floor(y);
  w = 0;

  if (m >= 0 && m < H && n >= 0 && n < W) {
    // 0 <= m < H and 0 <= n < W mean left_top point inside image
    w = std::max((float)0, 1 - std::abs(x - m)) * std::max((float)0, 1 - std::abs(y - n));
    res += w * pic[m * W + n];
  }

  m = std::floor(x) + 1;
  n = std::floor(y);
  w = 0;

  if (m >= 0 && m < H && n >= 0 && n < W) {
    w = std::max((float)0, 1 - std::abs(x - m)) * std::max((float)0, 1 - std::abs(y - n));
    res += w * pic[m * W + n];
  }

  m = std::floor(x);
  n = std::floor(y) + 1;
  w = 0;

  if (m >= 0 && m < H && n >= 0 && n < W) {
    w = std::max((float)0, 1 - std::abs(x - m)) * std::max((float)0, 1 - std::abs(y - n));
    res += w * pic[m * W + n];
  }

  m = std::floor(x) + 1;
  n = std::floor(y) + 1;
  w = 0;

  if (m >= 0 && m < H && n >= 0 && n < W) {
    w = std::max((float)0, 1 - std::abs(x - m)) * std::max((float)0, 1 - std::abs(y - n));
    res += w * pic[m * W + n];
  }
  return (float)res;
}

static void CpuPluginSpatialTransform(float *dst,
                                      float *src,
                                      float *mat,
                                      float *multable_value,
                                      int batch_size,
                                      int dst_h,
                                      int dst_w,
                                      int src_h,
                                      int src_w,
                                      int c,
                                      int mat_no_broadcast) {
  CHECK_VALID(dst);
  CHECK_VALID(src);
  CHECK_VALID(mat);
  for (int i = 0; i < batch_size; i++) {
    // Update mat with multable value
    // mat's shape must be [N 6], multable_value's shape must be [N 2]
    mat[0 + i * 6 * mat_no_broadcast] = multable_value[0 + i * 2];
    mat[4 + i * 6 * mat_no_broadcast] = multable_value[1 + i * 2];
    float *affined_ptr = dst + i * dst_h * dst_w * c;
    float *src_ptr = src + i * src_h * src_w * c;

    for (int _h = 0; _h < dst_h; ++_h) {
      for (int _w = 0; _w < dst_w; ++_w) {
        float w_ = ((float)_w / (float)dst_w) * 2 - 1;
        float h_ = ((float)_h / (float)dst_h) * 2 - 1;

        float reflect_h = w_ * mat[0 + i * 6 * mat_no_broadcast] +
                          h_ * mat[1 + i * 6 * mat_no_broadcast] +
                          mat[2 + i * 6 * mat_no_broadcast];
        float reflect_w = w_ * mat[3 + i * 6 * mat_no_broadcast] +
                          h_ * mat[4 + i * 6 * mat_no_broadcast] +
                          mat[5 + i * 6 * mat_no_broadcast];

        reflect_w = ((reflect_w + 1) / 2) * src_w;
        reflect_h = ((reflect_h + 1) / 2) * src_h;

        for (int k = 0; k < c; ++k) {
          affined_ptr[(_h * dst_w + _w) * c + k] =
              SpatialTransformCpuForward(src_ptr, reflect_h, reflect_w, src_h, src_w);
        }
      }
    }
  }
}

class CPUSpatialTransArg : public ArgListBase {
  DECLARE_ARG(layout, (std::string))
      ->SetDescription("Input/Output layout")
      ->SetAlternative({"NCHW", "NHWC"});
  DECLARE_ARG(datatype, (std::string))
      ->SetDescription("Input/Output datatype")
      ->SetAlternative({"float", "half"});
  DECLARE_ARG(input_batches, (std::vector<int>))
      ->SetDescription("Input batches for cpu_spatial_trans.");
  DECLARE_ARG(input_dir, (std::string))->SetDescription("Input dir for cpu_spatial_trans.");
};

int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  CPUSpatialTransArg arg_reader;
  arg_reader.ReadIn(args);
  auto layout = Value(arg_reader.layout());
  auto datatype = Value(arg_reader.datatype());
  auto input_batches = Value(arg_reader.input_batches());
  auto input_dir = Value(arg_reader.input_dir());

  /*
   *  in/out hw must be 40, 180
   * and channel must be 1
   * So in nhwc, shapes are like:
   * input:batch[0], 40, 180, 1\
   * mat:batch[1], 6, 1, 1 (or batch[1], 6)
   * mutable:batch[2], 2, 1, 1 (or batch[2], 2)
   */
  CHECK_EQ(input_batches.size(), 3);
  CHECK_EQ(input_batches[0], input_batches[2]);
  int no_broadcast = 1;
  if (input_batches[1] == 1) {
    no_broadcast = 0;
  } else {
    CHECK_EQ(input_batches[0], input_batches[1]);
  }
  std::vector<float> input(input_batches[0] * 40 * 180);
  std::vector<float> mat(input_batches[1] * 6);
  std::vector<float> muta(input_batches[2] * 2);
  CHECK_VALID(
      ReadDataFromFile(input_dir + "/input1_fp32.bin", input.data(), input.size() * sizeof(float)));
  CHECK_VALID(
      ReadDataFromFile(input_dir + "/input2_fp32.bin", mat.data(), mat.size() * sizeof(float)));
  CHECK_VALID(
      ReadDataFromFile(input_dir + "/input3_fp32.bin", muta.data(), muta.size() * sizeof(float)));
  std::vector<float> output(input_batches[0] * 40 * 180);
  if (datatype == "half") {
    // cvt float-half-float to simulate half cast precision loss
    std::vector<half_float::half> input_half(input_batches[0] * 40 * 180);
    std::vector<half_float::half> mat_half(input_batches[1] * 6);
    std::vector<half_float::half> muta_half(input_batches[2] * 2);
    std::vector<half_float::half> output_half(input_batches[0] * 40 * 180);

    CHECK_STATUS(NormalCast(input_half.data(), magicmind::DataType::FLOAT16, input.data(),
                            magicmind::DataType::FLOAT32, input.size(), false));
    CHECK_VALID(WriteDataToFile("./input_data", input_half.data(),
                                input_half.size() * sizeof(half_float::half)));
    CHECK_STATUS(NormalCast(input.data(), magicmind::DataType::FLOAT32, input_half.data(),
                            magicmind::DataType::FLOAT16, input.size(), false));

    CHECK_STATUS(NormalCast(mat_half.data(), magicmind::DataType::FLOAT16, mat.data(),
                            magicmind::DataType::FLOAT32, mat.size(), false));
    CHECK_VALID(
        WriteDataToFile("./mat_data", mat_half.data(), mat_half.size() * sizeof(half_float::half)));
    CHECK_STATUS(NormalCast(mat.data(), magicmind::DataType::FLOAT32, mat_half.data(),
                            magicmind::DataType::FLOAT16, mat.size(), false));

    CHECK_STATUS(NormalCast(muta_half.data(), magicmind::DataType::FLOAT16, muta.data(),
                            magicmind::DataType::FLOAT32, muta.size(), false));
    CHECK_VALID(WriteDataToFile("./muta_data", muta_half.data(),
                                muta_half.size() * sizeof(half_float::half)));
    CHECK_STATUS(NormalCast(muta.data(), magicmind::DataType::FLOAT32, muta_half.data(),
                            magicmind::DataType::FLOAT16, muta.size(), false));
    CpuPluginSpatialTransform(output.data(), input.data(), mat.data(), muta.data(),
                              input_batches[0], 40, 180, 40, 180, 1, no_broadcast);
    CHECK_STATUS(NormalCast(output_half.data(), magicmind::DataType::FLOAT16, output.data(),
                            magicmind::DataType::FLOAT32, output.size(), false));
    CHECK_VALID(WriteDataToFile("./baseline", output_half.data(),
                                output_half.size() * sizeof(half_float::half)));
  } else {
    CHECK_VALID(WriteDataToFile("./input_data", input.data(), input.size() * sizeof(float)));
    CHECK_VALID(WriteDataToFile("./mat_data", mat.data(), mat.size() * sizeof(float)));
    CHECK_VALID(WriteDataToFile("./muta_data", muta.data(), muta.size() * sizeof(float)));
    CpuPluginSpatialTransform(output.data(), input.data(), mat.data(), muta.data(),
                              input_batches[0], 40, 180, 40, 180, 1, no_broadcast);
    CHECK_VALID(WriteDataToFile("./baseline", output.data(), output.size() * sizeof(float)));
  }
  return 0;
}
