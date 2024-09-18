# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.

cmake_minimum_required(VERSION 3.5)

project(basic_samples)

set(MAGICMIND_SAMPLE ${MAGICMIND_SAMPLE})
message("-- MAGICMIND_SAMPLE=${MAGICMIND_SAMPLE}")
set(MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP ${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP})
message("-- MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP=${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP}")
set(MAGICMIND_PLUGIN_CROP_SAMPLE ${MAGICMIND_PLUGIN_CROP_SAMPLE})
message("-- MAGICMIND_PLUGIN_CROP_SAMPLE=${MAGICMIND_PLUGIN_CROP_SAMPLE}")
list(LENGTH MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP plugin_length)
if (${plugin_length} GREATER 0)
  set(BUILD_WITH_BANG ON)
endif()
set(MAGICMIND_OPS_SAMPLE ${MAGICMIND_OPS_SAMPLE})
message("-- MAGICMIND_OPS_SAMPLE=${MAGICMIND_OPS_SAMPLE}")

include("../CMakeSampleTemplate.txt")

include_directories(${PROJECT_SOURCE_DIR})
###############################################################################
# Basic funcs
###############################################################################
function(add_executable_sample target)
  set(flags)
  set(args)
  set(listArgs SRCS DEPS)
  cmake_parse_arguments(arg "${flags}" "${args}" "${listArgs}" ${ARGN})
  foreach(sample ${MAGICMIND_SAMPLE})
    if ("sample_${sample}" STREQUAL target)
      if (NOT arg_SRCS)
        file(GLOB files ${target}/*.cc)
      else()
        set(files ${arg_SRCS})
      endif()
      if ("${sample}" STREQUAL "calibration")
        add_calib_obj()
      endif()
      add_executable(${target} ${files})
      message(STATUS "To compile ${target} with ${arg_DEPS}")
      target_link_libraries(${target} PRIVATE ${arg_DEPS})
      add_custom_command(
          TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${PROJECT_SOURCE_DIR}/${target}/run.sh
                  ${CMAKE_CURRENT_BINARY_DIR}/run_${target}.sh)
    endif()
  endforeach()
endfunction()

function(add_pluginop_sample op)
  message(STATUS "Pluginop sample: ${op}")
  add_executable("sample_${op}" "${plugin_sample_path}/Plugin${op}/build.cc")
  target_link_libraries("sample_${op}" PRIVATE magicmind cnrt common_obj_runtime dl)
  add_executable("cpu_${op}" "${plugin_sample_path}/Plugin${op}/cpu_impl.cc")
  target_link_libraries("cpu_${op}" PRIVATE common_obj_runtime)
  add_custom_command(
    TARGET cpu_${op} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${PROJECT_SOURCE_DIR}/sample_pluginop/samples/Plugin${op}/run.sh
            ${CMAKE_CURRENT_BINARY_DIR}/run_plugin_${op}_op.sh)
endfunction()

function(add_op_sample target)
  foreach(sample ${MAGICMIND_OPS_SAMPLE})
    if ("sample_${sample}" STREQUAL target)
      if (NOT arg_SRCS)
        file(GLOB files ${sample_ops_path}/${target}/*.cc)
      else()
        set(files ${arg_SRCS})
      endif()
      add_executable(${target} ${files})
      message(STATUS "To compile ${target}")
      #target_link_libraries(${target} PRIVATE ${arg_DEPS})
      target_link_libraries("${target}" PRIVATE magicmind cnrt common_obj_runtime dl)
      add_custom_command(
          TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${PROJECT_SOURCE_DIR}/sample_ops/${target}/run.sh
                  ${CMAKE_CURRENT_BINARY_DIR}/run_${target}.sh)
    endif()
  endforeach()
endfunction()

function(add_plugin_crop_sample target)
  add_executable("${target}" "${sample_plugin_crop_path}/build.cc")
  target_link_libraries("${target}" PRIVATE magicmind cnrt common_obj_runtime dl)
  add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${PROJECT_SOURCE_DIR}/sample_plugin_crop/run.sh
      ${CMAKE_CURRENT_BINARY_DIR}/run_plugin_crop.sh)
endfunction()
###############################################################################
# Basic samples
###############################################################################
#always compile remote server
add_executable(remote_server sample_runtime/remote_server.cc)
target_link_libraries(remote_server PRIVATE magicmind_runtime cnlight common_obj_runtime)
add_executable_sample(sample_calibration DEPS common_obj_runtime common_obj_compile magicmind magicmind_parser)
add_executable_sample(sample_quantization DEPS common_obj_runtime magicmind)
add_executable_sample(sample_network_modify DEPS common_obj_runtime magicmind)
add_executable_sample(sample_refit DEPS common_obj_runtime magicmind magicmind_runtime cnrt pthread)
add_executable_sample(sample_runtime SRCS sample_runtime/sample_runtime.cc DEPS common_obj_runtime magicmind_runtime cnlight cnrt pthread)
if (TARGET sample_calibration)
  add_custom_command(
    TARGET sample_calibration POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${PROJECT_SOURCE_DIR}/sample_calibration/sample_labels.txt
            ${CMAKE_CURRENT_BINARY_DIR}/sample_labels.txt)
endif()
###############################################################################
# Plugin
###############################################################################
set_ifndef(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
message("-- PLUGIN_LIBRARY_OUTPUT_PATH=${LIBRARY_OUTPUT_PATH}")
set(plugin_kernel_path ${PROJECT_SOURCE_DIR}/sample_pluginop/plugin_kernels)
set(plugin_sample_path ${PROJECT_SOURCE_DIR}/sample_pluginop/samples)

#Ignore bang cncc minor warnings
set(BANG_CNCC_FLAGS ${BANG_CNCC_FLAGS} "-Wno-unused-command-line-argument")
message("-- BANG_CNCC_FLAGS: ${BANG_CNCC_FLAGS}")

#Generate the required object files
foreach(op ${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP})
  if (EXISTS "${plugin_kernel_path}/Plugin${op}")
    #Get the host files and bang files of each plugin 
    file(GLOB_RECURSE host_src_files "${plugin_kernel_path}/Plugin${op}/*.cc")
    file(GLOB_RECURSE bang_src_files "${plugin_kernel_path}/Plugin${op}/*.mlu")
    #Make cncc to recognize the host files
    set_source_files_properties(${host_src_files} PROPERTIES BANG_SOURCE_PROPERTY_FORMAT "OBJ")
    #Use bang_warp_srcs target to generate the required object files
    bang_wrap_srcs(Plugin${op} OBJ ${op}_cpu_objs "${host_src_files}")
    bang_wrap_srcs(Plugin${op} OBJ ${op}_mlu_objs "${bang_src_files}")
    #Save the object files to vals
    set(total_cpu_objs  ${total_cpu_objs} ${${op}_cpu_objs})
    set(total_mlu_objs  ${total_mlu_objs} ${${op}_mlu_objs})
  else()
    message(FATAL_ERROR "${op} file doesn't exist.")
  endif()
endforeach()
message("-- total_mlu_objs=${total_mlu_objs}")
message("-- total_cpu_objs=${total_cpu_objs}")
#Generate the shared lib and static lib for plugin ops
if (${BUILD_WITH_BANG})
  #Use the object files to generate shared lib
  bang_add_library(magicmind_plugin SHARED "${total_mlu_objs}" "${total_cpu_objs}")
  set_target_properties(magicmind_plugin PROPERTIES LINKER_LANGUAGE "CXX")
  #Use the object files to generate static lib
  bang_add_library(magicmind_plugin_static STATIC "${total_mlu_objs}" "${total_cpu_objs}")
  set_target_properties(magicmind_plugin_static PROPERTIES LINKER_LANGUAGE "CXX")
  target_link_libraries(magicmind_plugin cnrt cnnl cnnl_extra magicmind_runtime common_obj_runtime)
  add_custom_command(
    TARGET magicmind_plugin POST_BUILD
    COMMAND ln -s ${PROJECT_SOURCE_DIR}/sample_pluginop/
            ${CMAKE_CURRENT_BINARY_DIR}/)
endif()

if (NOT EXISTS ${CMAKE_TOOLCHAIN_FILE})
  foreach(op ${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP})
    if (EXISTS "${plugin_sample_path}/Plugin${op}")
      add_pluginop_sample("${op}")
    else()
      message(FATAL_ERROR "${op} sample doesn't exist.")
    endif()
  endforeach()
endif()
###############################################################################
# Op samples
###############################################################################
set(sample_ops_path ${PROJECT_SOURCE_DIR}/sample_ops/)
foreach(op ${MAGICMIND_OPS_SAMPLE})
  if (EXISTS "${sample_ops_path}/sample_${op}")
    add_op_sample("sample_${op}")
  else()
    message(FATAL_ERROR "sample_${op} doesn't exist.")
  endif()
endforeach()
###############################################################################
# PluginCrop samples
###############################################################################
if (NOT EXISTS ${CMAKE_TOOLCHAIN_FILE})
  if ("${MAGICMIND_PLUGIN_CROP_SAMPLE}" STREQUAL "plugin_crop")
    set(sample_plugin_crop_path ${PROJECT_SOURCE_DIR}/sample_${MAGICMIND_PLUGIN_CROP_SAMPLE})
    if (EXISTS "${sample_plugin_crop_path}")
      add_plugin_crop_sample("sample_${MAGICMIND_PLUGIN_CROP_SAMPLE}")
    else()
      message(FATAL_ERROR "${MAGICMIND_PLUGIN_CROP_SAMPLE} sample dosen't exist.")
    endif()
  endif()
endif()

