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
  Dims input_dims = Dims({-1, -1, -1});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  // create mask tensor
  DataType mask_dtype = DataType::FLOAT32;
  Dims mask_dims = Dims({-1, -1});
  magicmind::ITensor *mask_tensor = network->AddInput(mask_dtype, mask_dims);
  CHECK_VALID(mask_tensor);

  // create weight tensor vector
  DataType weight_dtype = DataType::FLOAT32;
  Dims weight_ih_dims = Dims({-1, -1});
  Dims weight_hh_dims = Dims({-1, -1});
  Dims weight_ho_dims = Dims({-1, -1});
  magicmind::ITensor *weight_ih_tensor =
      network->AddInput(weight_dtype, weight_ih_dims);
  magicmind::ITensor *weight_hh_tensor =
      network->AddInput(weight_dtype, weight_hh_dims);
  magicmind::ITensor *weight_ho_tensor =
      network->AddInput(weight_dtype, weight_ho_dims);
  CHECK_VALID(weight_ih_tensor);
  CHECK_VALID(weight_hh_tensor);
  CHECK_VALID(weight_ho_tensor);
  std::vector<magicmind::ITensor *> weight_tensors(
      {weight_ih_tensor, weight_hh_tensor, weight_ho_tensor});

  // create bias tensor vector
  DataType bias_dtype = DataType::FLOAT32;
  Dims bias_ih_dims = Dims({-1});
  Dims bias_ho_dims = Dims({-1});
  magicmind::ITensor *bias_ih_tensor =
      network->AddInput(bias_dtype, bias_ih_dims);
  magicmind::ITensor *bias_ho_tensor =
      network->AddInput(bias_dtype, bias_ho_dims);
  CHECK_VALID(bias_ih_tensor);
  CHECK_VALID(bias_ho_tensor);
  std::vector<magicmind::ITensor *> bias_tensors(
      {bias_ih_tensor, bias_ho_tensor});

  // create rnn node
  IRNNNode *RNN = network->AddIRNNNode(input_tensor, mask_tensor, nullptr,
                                       weight_tensors, bias_tensors);

  // set rnn attribution
  CHECK_STATUS(RNN->SetHasMask(true));
  CHECK_STATUS(RNN->SetSkipInput(false));
  CHECK_STATUS(RNN->SetBidirectional(false));
  CHECK_STATUS(RNN->SetBiasMode(IRNNBiasMode::NO_BIAS));
  CHECK_STATUS(RNN->SetNumLayers(1));
  CHECK_STATUS(RNN->SetInputSize(374));
  CHECK_STATUS(RNN->SetHiddenSize(128));
  CHECK_STATUS(RNN->SetExposeHidden(false));
  CHECK_STATUS(RNN->SetOutputMode(IRNNOutputMode::HAS_OUT_LAYER));
  CHECK_STATUS(RNN->SetActiveMode(IActivation::TANH));
  CHECK_STATUS(RNN->SetLayout(Layout::TNC, Layout::TN, Layout::TNC));

  // mark network output
  for (auto i = 0; i < RNN->GetOutputCount(); i++) {
    auto output_tensor = RNN->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(
      builder->BuildModel("rnn_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("rnn_model"));

  return 0;
}
