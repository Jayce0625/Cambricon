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
  // input is variadic, please giving correct number of input's dtype and dims.
  std::vector<DataType> data_dtype = {DataType::FLOAT32};
  std::vector<Dims> data_dims = {Dims({-1, -1})};

  std::vector<ITensor *> data_tensors;
  for (auto i = 0; i < data_dims.size(); i++) {
    ITensor *temp_tensor = network->AddInput(data_dtype[i], data_dims[i]);
    CHECK_VALID(temp_tensor);
    data_tensors.push_back(temp_tensor);
  }

  std::vector<int32_t> indice_value = {0,1,4,2,3,5};
  magicmind::IConstNode *indice = network->AddIConstNode(DataType::INT32, Dims({6}), indice_value.data());
  CHECK_VALID(indice);
  
  // create dynamic_stitch node
  // IDynamicStitchNode *DynamicStitch = network->AddIDynamicStitchNode(data_tensors, indices_tensors);
  IDynamicStitchNode *DynamicStitch = network->AddIDynamicStitchNode(data_tensors, {indice->GetOutput(0)});

  // mark network output
  for (auto i = 0; i < DynamicStitch->GetOutputCount(); i++) {
    auto output_tensor = DynamicStitch->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("dynamic_stitch_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("dynamic_stitch_model"));

  return 0;
}
