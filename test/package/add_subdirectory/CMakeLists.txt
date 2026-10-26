#
# SPDX-License-Identifier: MPL-2.0
#
# add_subdirectory consumer: the aV2P source tree is part of the consumer build
# and the aV2P::aV2P target is used directly. No installation is involved.
#
cmake_minimum_required(VERSION 3.25)
project(aV2PAddSubdirectoryConsumer LANGUAGES CXX)

if(NOT AV2P_SOURCE_DIR)
    message(FATAL_ERROR "AV2P_SOURCE_DIR must point to the aV2P source tree")
endif()

set(aV2P_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(aV2P_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "")

add_subdirectory("${AV2P_SOURCE_DIR}" aV2P)

add_executable(aV2P-add-subdirectory main.cpp)
target_link_libraries(aV2P-add-subdirectory PRIVATE aV2P::aV2P)
# Native alpaka 3 requires finalizing every target that links it.
aV2P_finalize(aV2P-add-subdirectory)
