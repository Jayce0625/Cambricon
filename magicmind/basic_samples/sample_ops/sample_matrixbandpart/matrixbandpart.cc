/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "mm_builder.h"  // NOLINT
#include "mm_network.h"  // NOLINT
#include "mm_runtime.h"  // NOLINT
#include "common/macros.h"
#include "common/data.h"
#include "common/container.h"
using namespace magicmind;  // NOLINT
int main(int argc, char **argv) {
  // init
  auto builder = SUniquePtr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  CHECK_VALID(builder);
  auto network = SUniquePtr<magicmind::INetwork>(magicmind::CreateINetwork());
  CHECK_VALID(network);

  // create input tensor
  DataType input_dtype = DataType::FLOAT32;
  Dims input_dims = Dims({-1, -1, -1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  const std::vector<int64_t> scalar_dimensions;
  DataType num_lower_dtype = DataType::INT64;
  Dims num_lower_dims = Dims(scalar_dimensions);
  magicmind::ITensor *num_lower_tensor = network->AddInput(num_lower_dtype, num_lower_dims);
  CHECK_VALID(num_lower_tensor);

  DataType num_upper_dtype = DataType::INT64;
  Dims num_upper_dims = Dims(scalar_dimensions);
  magicmind::ITensor *num_upper_tensor = network->AddInput(num_upper_dtype, num_upper_dims);
  CHECK_VALID(num_upper_tensor);

  // create matrix_band_part node
  IMatrixBandPartNode *MatrixBandPart = network->AddIMatrixBandPartNode(
    input_tensor, num_lower_tensor, num_upper_tensor);

  // mark network output
  for (auto i = 0; i < MatrixBandPart->GetOutputCount(); i++) {
    auto output_tensor = MatrixBandPart->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(
    builder->BuildModel("matrix_band_part_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("matrixbandpart_model"));

  return 0;
}
