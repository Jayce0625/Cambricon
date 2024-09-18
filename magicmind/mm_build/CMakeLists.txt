# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

cmake_minimum_required(VERSION 3.5)

project(mm_build)

include("../CMakeSampleTemplate.txt")

include_directories(${PROJECT_SOURCE_DIR})

add_library(build_obj OBJECT ./main_process.cc ./parser.cc)

add_calib_obj()

function(add_executable_with name)
  message(STATUS "compile ${name}")
  string(TOUPPER ${name} _NAME)
  add_executable(${name}_build $<TARGET_OBJECTS:build_obj> ./main.cc)
  target_compile_definitions(${name}_build PRIVATE BUILD_${_NAME})
  target_link_libraries(${name}_build common_obj_compile common_obj_runtime cnrt magicmind magicmind_parser magicmind_runtime pthread dl)
endfunction(add_executable_with name)

add_executable_with(caffe)
add_executable_with(onnx)
add_executable_with(tensorflow)
add_executable_with(pytorch)
add_custom_command(
  TARGET pytorch_build POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${PROJECT_SOURCE_DIR}/run.sh
          ${CMAKE_CURRENT_BINARY_DIR}/run.sh)
