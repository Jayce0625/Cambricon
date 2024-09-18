/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Host implement for resize
 *************************************************************************/
#include "common/data.h"
#include "common/macros.h"
static void ResizeCpuCompute(unsigned char *dst,
                             unsigned char *src,
                             int s_row,
                             int s_col,
                             int d_row,
                             int d_col) {
  int height = s_row;
  int width = s_col;
  double scale_x = (double)width / d_col;
  double scale_y = (double)height / d_row;
  for (int i = 0; i < d_row; i++) {
    double map_y = (i + 0.5) * scale_y - 0.5;
    int yi_0 = (int)map_y;
    int yi_1 = yi_0 + 1;
    double yd = map_y - yi_0;
    if (map_y < 0) {
      map_y = 0;
      yi_0 = (int)map_y;
      yi_1 = yi_0 + 1;
      yd = 0;
    }
    if (map_y >= s_row - 1) {
      map_y = s_row - 2;
      yi_0 = (int)map_y;
      yi_1 = yi_0 + 1;
      yd = 1;
    }

    for (int j = 0; j < d_col; j++) {
      double map_x = (j + 0.5) * scale_x - 0.5;
      int xi_0 = (int)map_x;
      int xi_1 = xi_0 + 1;
      double xd = map_x - xi_0;
      if (map_x < 0) {
        map_x = 0;
        xi_0 = (int)map_x;
        xi_1 = xi_0 + 1;
        xd = 0;
      }
      if (map_x >= s_col - 1) {
        map_x = s_col - 2;
        xi_0 = (int)map_x;
        xi_1 = xi_0 + 1;
        xd = 1;
      }

      for (int k = 0; k < 4; k++) {
        unsigned char index00 = src[yi_0 * width * 4 + xi_0 * 4 + k];
        unsigned char index01 = src[yi_0 * width * 4 + xi_1 * 4 + k];
        unsigned char index10 = src[yi_1 * width * 4 + xi_0 * 4 + k];
        unsigned char index11 = src[yi_1 * width * 4 + xi_1 * 4 + k];

        dst[i * d_col * 4 + j * 4 + k] = (1 - xd) * (1 - yd) * index00 + xd * (1 - yd) * index01 +
                                         (1 - xd) * yd * index10 + xd * yd * index11;
      }
    }
  }
}

static void CropAndResizeCpuCompute(uint8_t *cpu_output_ptr,
                                    uint8_t *cpu_input_ptr,
                                    int *cpu_crop_params_ptr,
                                    int *cpu_roi_nums_ptr,
                                    int *cpu_pad_values_ptr,
                                    int s_row,
                                    int s_col,
                                    int d_row,
                                    int d_col,
                                    int batch_size,
                                    int keep_aspect_ratio) {
  uint8_t *tmp_cpu_output = (uint8_t *)malloc(d_row * d_col * 4 * sizeof(uint8_t));
  uint8_t *tmp_cpu_input = (uint8_t *)malloc(s_row * s_col * 4 * sizeof(uint8_t));
  int *tmp_crop_params = cpu_crop_params_ptr;
  int roi_count = 0;
  int dst_batch_size = d_row * d_col * 4;
  for (int i = 0; i < batch_size; i++) {
    int batch_roi_nums = cpu_roi_nums_ptr[i];
    for (int j = 0; j < batch_roi_nums; j++) {
      int roi_x = tmp_crop_params[0];
      int roi_y = tmp_crop_params[1];
      int roi_w = tmp_crop_params[2];
      int roi_h = tmp_crop_params[3];

      for (int row = 0; row < roi_h; row++) {
        for (int col = 0; col < roi_w; col++) {
          for (int ch = 0; ch < 4; ch++) {
            int offset_in =
                i * s_col * s_row * 4 + (row + roi_y) * s_col * 4 + (col + roi_x) * 4 + ch;
            int offset_tp = row * roi_w * 4 + col * 4 + ch;
            tmp_cpu_input[offset_tp] = cpu_input_ptr[offset_in];
          }
        }
      }

      for (int ii = 0; ii < d_row * d_col; ii++) {
        cpu_output_ptr[dst_batch_size * roi_count + ii * 4 + 0] = (uint8_t)cpu_pad_values_ptr[0];
        cpu_output_ptr[dst_batch_size * roi_count + ii * 4 + 1] = (uint8_t)cpu_pad_values_ptr[1];
        cpu_output_ptr[dst_batch_size * roi_count + ii * 4 + 2] = (uint8_t)cpu_pad_values_ptr[2];
        cpu_output_ptr[dst_batch_size * roi_count + ii * 4 + 3] = (uint8_t)cpu_pad_values_ptr[3];
      }

      int d_row_ar = d_row;
      int d_col_ar = d_col;
      if (keep_aspect_ratio) {
        float src_ar = (float)(roi_w) / roi_h;
        float dst_ar = (float)(d_col) / d_row;
        if (src_ar > dst_ar) {
          d_row_ar = d_col * roi_h / roi_w;
        } else {
          d_col_ar = d_row * roi_w / roi_h;
        }
      }

      ResizeCpuCompute(tmp_cpu_output, tmp_cpu_input, roi_h, roi_w, d_row_ar, d_col_ar);

      for (int row = 0; row < d_row_ar; row++) {
        for (int col = 0; col < d_col_ar; col++) {
          for (int ch = 0; ch < 4; ch++) {
            int offset_out = roi_count * d_col * d_row * 4 + d_col * row * 4 + col * 4 + ch;
            int offset_tmp = d_col_ar * row * 4 + col * 4 + ch;
            cpu_output_ptr[offset_out] = tmp_cpu_output[offset_tmp];
          }
        }
      }
      tmp_crop_params += 4;
      roi_count += 1;
    }
  }
  free(tmp_cpu_input);
  free(tmp_cpu_output);
}

// To gen input and comput output
int main() {
  // params
  int batch_size = 4;
  int channel = 4;
  int s_row = 1080;
  int s_col = 608;
  int xywh = 4;
  int64_t d_row = 200;
  int64_t d_col = 200;
  int64_t keep_aspect_ratio = 1;
  // input
  unsigned int input_num = batch_size * channel * s_row * s_col;
  std::vector<uint8_t> input = GenRand<uint8_t>(input_num, 0, 255, 0);
  // rois
  int total_rois = 0;
  unsigned int roi_nums_num = batch_size;
  std::vector<int> roi_nums(roi_nums_num);
  for (int i = 0; i < batch_size; i++) {
    roi_nums[i] = i + 1;
    total_rois += i + 1;
  }
  // crop param
  unsigned int crop_params_num = total_rois * xywh;
  std::vector<int> crop_params(crop_params_num);
  for (int i = 0; i < total_rois; i++) {
    crop_params[4 * i + 0] = i;
    crop_params[4 * i + 1] = i;
    crop_params[4 * i + 2] = s_col - i;
    crop_params[4 * i + 3] = s_row - i;
  }
  // pad_value
  std::vector<int> pad_values{1, 2, 3, 4};

  unsigned int output_num = total_rois * channel * d_col * d_row;
  std::vector<uint8_t> output(output_num);
  CropAndResizeCpuCompute(output.data(), input.data(), crop_params.data(), roi_nums.data(),
                          pad_values.data(), s_row, s_col, d_row, d_col, batch_size,
                          keep_aspect_ratio);
  // write out for future comparasion to dev
  CHECK_VALID(WriteDataToFile("./input", input.data(), input_num * sizeof(uint8_t)));
  CHECK_VALID(WriteDataToFile("./roi_nums", roi_nums.data(), roi_nums_num * sizeof(int)));
  CHECK_VALID(WriteDataToFile("./crop_params", crop_params.data(), crop_params_num * sizeof(int)));
  CHECK_VALID(WriteDataToFile("./pad_values", pad_values.data(), pad_values.size() * sizeof(int)));
  CHECK_VALID(WriteDataToFile("./baseline", output.data(), output.size() * sizeof(uint8_t)));
}
