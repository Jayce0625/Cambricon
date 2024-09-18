/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLE_REFIT_MODEL_H
#define SAMPLE_REFIT_MODEL_H

#include "mm_builder.h"
#include "mm_runtime.h"

/**
 * A simple conv model
 */
magicmind::IModel *CreateModel(magicmind::IBuilderConfig *builder_cfg);
#endif  // SAMPLE_REFIT_MODEL_H
