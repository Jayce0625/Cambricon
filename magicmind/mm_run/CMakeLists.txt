# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

cmake_minimum_required(VERSION 3.5)

project(mm_run)

include("../CMakeSampleTemplate.txt")
include_directories(${PROJECT_SOURCE_DIR})

add_library(run_obj OBJECT ./inference.cc ./run.cc ./shape_groups.cc ./stage.cc ./trace.cc)
target_compile_definitions(run_obj PRIVATE -DUSE_PROFILER)

message(STATUS "compile mm_run")
add_executable(mm_run $<TARGET_OBJECTS:run_obj> ./main.cc)

target_compile_definitions(mm_run PRIVATE -DUSE_PROFILER)
target_link_libraries(mm_run common_obj_runtime cnrt magicmind_runtime cnlight pthread dl)
