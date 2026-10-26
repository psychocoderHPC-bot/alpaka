#
# SPDX-License-Identifier: MPL-2.0
#
# Negative fixture: a target created with plain add_executable() that links
# aV2P::aV2P but never calls aV2P_finalize(). Configuration must fail with a
# diagnostic naming the target and pointing at aV2P_finalize().
#
cmake_minimum_required(VERSION 3.25)
project(aV2PUnfinalizedFixture LANGUAGES CXX)

if(NOT AV2P_SOURCE_DIR)
    message(FATAL_ERROR "AV2P_SOURCE_DIR must point to the aV2P source tree")
endif()

set(aV2P_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(aV2P_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "")

add_subdirectory("${AV2P_SOURCE_DIR}" aV2P)

add_executable(aV2P-unfinalized main.cpp)
target_link_libraries(aV2P-unfinalized PRIVATE aV2P::aV2P)
