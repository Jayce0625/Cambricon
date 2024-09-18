/*************************************************************************
* Copyright (C) [2020-2023] by Cambricon, Inc.
*************************************************************************/
#include <iostream>
#include <memory>
#include <vector>
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
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

  DataType begin_dtype = DataType::INT32;
  Dims begin_dims = Dims({-1});
  magicmind::ITensor *begin_tensor = network->AddInput(begin_dtype, begin_dims);
  CHECK_VALID(begin_tensor);

  DataType end_dtype = DataType::INT32;
  Dims end_dims = Dims({-1});
  magicmind::ITensor *end_tensor = network->AddInput(end_dtype, end_dims);
  CHECK_VALID(end_tensor);

  // optional input
  DataType strides_dtype = DataType::INT32;
  Dims strides_dims = Dims({-1});
  magicmind::ITensor *strides_tensor = network->AddInput(strides_dtype, strides_dims);
  CHECK_VALID(strides_tensor);

  // optional input
  std::vector<int32_t> axis_value = {0,1,2,3};
  magicmind::IConstNode *axis = network->AddIConstNode(DataType::INT32, Dims({4}), axis_value.data());
  CHECK_VALID(axis);

  // create strided_slice node
  IStridedSliceNode *StridedSlice = network->AddIStridedSliceNode(input_tensor,begin_tensor,end_tensor,strides_tensor,axis->GetOutput(0));

  //using StridedSlice default paramters, you can set each attribute's value.
  //CHECK_STATUS(StridedSlice->SetBeginMask([value]));
  //CHECK_STATUS(StridedSlice->SetEndMask([value]));
  //CHECK_STATUS(StridedSlice->SetEllipsisMask([value]));
  //CHECK_STATUS(StridedSlice->SetNewAxisMask([value]));
  //CHECK_STATUS(StridedSlice->SetShrinkAxisMask([value]));
  //CHECK_STATUS(StridedSlice->SetShapeinferZero([value]));

  // mark network output
  for (auto i = 0; i < StridedSlice->GetOutputCount(); i++) {
    auto output_tensor = StridedSlice->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("strided_slice_model", network.get()));
  CHECK_VALID(model);
  // save model to file
  CHECK_STATUS(model->SerializeToFile("strided_slice_model"));

  return 0;
}