# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

cmake_minimum_required(VERSION 3.5)

project(diff)

include("../../CMakeSampleTemplate.txt")

include_directories(${PROJECT_SOURCE_DIR})

add_executable(diff_compare ./diff.cc)

target_link_libraries(diff_compare PRIVATE common_obj_runtime)
