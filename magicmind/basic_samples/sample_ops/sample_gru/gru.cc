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
  Dims input_dims = Dims({-1, -1, 52});
  magicmind::ITensor *input_tensor = network->AddInput(input_dtype, input_dims);
  CHECK_VALID(input_tensor);

  auto filter_ih_dim = magicmind::Dims({10, 52});
  std::vector<float> filter_ih_r_buffer = GenRand(filter_ih_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *filter_ih_r = network->AddIConstNode(DataType::FLOAT32, filter_ih_dim, filter_ih_r_buffer.data());
  CHECK_VALID(filter_ih_r);

  std::vector<float> filter_ih_z_buffer = GenRand(filter_ih_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *filter_ih_z = network->AddIConstNode(DataType::FLOAT32, filter_ih_dim, filter_ih_z_buffer.data());
  CHECK_VALID(filter_ih_z);

  std::vector<float> filter_ih_n_buffer = GenRand(filter_ih_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *filter_ih_n = network->AddIConstNode(DataType::FLOAT32, filter_ih_dim, filter_ih_n_buffer.data());
  CHECK_VALID(filter_ih_n);

  auto filter_hh_dim = magicmind::Dims({10, 10});
  std::vector<float> filter_hh_r_buffer = GenRand(filter_hh_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *filter_hh_r = network->AddIConstNode(DataType::FLOAT32, filter_hh_dim, filter_hh_r_buffer.data());
  CHECK_VALID(filter_hh_r);

  std::vector<float> filter_hh_z_buffer = GenRand(filter_hh_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *filter_hh_z = network->AddIConstNode(DataType::FLOAT32, filter_hh_dim, filter_hh_z_buffer.data());
  CHECK_VALID(filter_hh_z);

  std::vector<float> filter_hh_n_buffer = GenRand(filter_hh_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *filter_hh_n = network->AddIConstNode(DataType::FLOAT32, filter_hh_dim, filter_hh_n_buffer.data());
  CHECK_VALID(filter_hh_n);

  // optional input
  auto bias_ih_dim = magicmind::Dims({10});
  std::vector<float> bias_ih_r_buffer = GenRand(bias_ih_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *bias_ih_r = network->AddIConstNode(DataType::FLOAT32, bias_ih_dim, bias_ih_r_buffer.data());
  CHECK_VALID(bias_ih_r);

  std::vector<float> bias_ih_z_buffer = GenRand(bias_ih_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *bias_ih_z = network->AddIConstNode(DataType::FLOAT32, bias_ih_dim, bias_ih_z_buffer.data());
  CHECK_VALID(bias_ih_z);

  std::vector<float> bias_ih_n_buffer = GenRand(bias_ih_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *bias_ih_n = network->AddIConstNode(DataType::FLOAT32, bias_ih_dim, bias_ih_n_buffer.data());
  CHECK_VALID(bias_ih_n);

  auto bias_hh_dim = magicmind::Dims({10});
  std::vector<float> bias_hh_r_buffer = GenRand(bias_hh_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *bias_hh_r = network->AddIConstNode(DataType::FLOAT32, bias_hh_dim, bias_hh_r_buffer.data());
  CHECK_VALID(bias_hh_r);

  std::vector<float> bias_hh_z_buffer = GenRand(bias_hh_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *bias_hh_z = network->AddIConstNode(DataType::FLOAT32, bias_hh_dim, bias_hh_z_buffer.data());
  CHECK_VALID(bias_hh_z);

  std::vector<float> bias_hh_n_buffer = GenRand(bias_hh_dim.GetElementCount(), -1.0f, 1.0f, 0);
  IConstNode *bias_hh_n = network->AddIConstNode(DataType::FLOAT32, bias_hh_dim, bias_hh_n_buffer.data());
  CHECK_VALID(bias_hh_n);

  // create gru node
  IGruNode *Gru = network->AddIGruNode(input_tensor, nullptr,
        {filter_ih_r->GetOutput(0), filter_ih_z->GetOutput(0), filter_ih_n->GetOutput(0),
         filter_hh_r->GetOutput(0), filter_hh_z->GetOutput(0), filter_hh_n->GetOutput(0)},
        {bias_ih_r->GetOutput(0), bias_ih_z->GetOutput(0), bias_ih_n->GetOutput(0),
         bias_hh_r->GetOutput(0), bias_hh_z->GetOutput(0), bias_hh_n->GetOutput(0)});

  //using Gru default paramters, you can set each attribute's value.
  //CHECK_STATUS(Gru->SetDirection([value]));
  //CHECK_STATUS(Gru->SetFilterOrder([value]));
  //CHECK_STATUS(Gru->SetGruAlgo([value]));
  CHECK_STATUS(Gru->SetNumLayers(1));
  CHECK_STATUS(Gru->SetInputSize(52));
  CHECK_STATUS(Gru->SetHiddenSize(10));
  //CHECK_STATUS(Gru->SetLayout([value]));

  // mark network output
  for (auto i = 0; i < Gru->GetOutputCount(); i++) {
    auto output_tensor = Gru->GetOutput(i);
    CHECK_STATUS(network->MarkOutput(output_tensor));
  }

  // create model
  auto model = SUniquePtr<magicmind::IModel>(builder->BuildModel("gru_model", network.get()));
  CHECK_VALID(model);

  // save model to file
  CHECK_STATUS(model->SerializeToFile("gru_model"));

  return 0;
}
