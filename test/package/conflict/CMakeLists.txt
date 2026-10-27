#
# SPDX-License-Identifier: MPL-2.0
#
# Negative fixture: aV2P_ACC_GPU_CUDA_ENABLE=ON with alpaka_DEP_CUDA=OFF must
# fail during configuration with a message naming both options.
#
cmake_minimum_required(VERSION 3.25)
project(aV2PConflictFixture LANGUAGES CXX)

if(NOT AV2P_SOURCE_DIR)
    message(FATAL_ERROR "AV2P_SOURCE_DIR must point to the aV2P source tree")
endif()

add_subdirectory("${AV2P_SOURCE_DIR}" aV2P)
