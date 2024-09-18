/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef SAMPLE_REFIT_WORKER_H
#define SAMPLE_REFIT_WORKER_H

#include <atomic>
#include "mm_runtime.h"
/**
 * Usually we will use some queue to load inputs and store outputs,
 * to simplify, we just generate inputs inplace and discard outputs.
 * To end the inference loop, we use a atomic bool as signal.
 */
void Worker(magicmind::IEngine *engine, int id, std::atomic<bool> *is_should_stop);

#endif  // SAMPLE_REFIT_WORKER_H
