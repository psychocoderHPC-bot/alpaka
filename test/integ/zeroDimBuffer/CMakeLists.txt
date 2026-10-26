#
# Copyright 2021 Andrea Bocci
# SPDX-License-Identifier: MPL-2.0
#

# aV2P 0D limitation: alpaka 3.0.0 has no zero-dimensional memory support, so every test case in
# src/zeroDimBuffer.cpp is inside `#if 0`. Building and registering the target anyway would execute
# zero tests and exit with status 0, which CTest reports as a pass. Do not build or register the
# target until zero-dimensional Buf/View/alloc support is restored together with the adapter.
set(_aV2P_ZERO_DIM_TESTS_ENABLED OFF)

# Also disable the test for CUDA and HIP only mode, as the test requires the serial CPU backend.
if(_aV2P_ZERO_DIM_TESTS_ENABLED
   AND NOT (aV2P_ACC_GPU_CUDA_ONLY_MODE OR aV2P_ACC_GPU_HIP_ONLY_MODE))
    set(_TARGET_NAME "zeroDimBufferTest")

    append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

    aV2P_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE})
    target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE common)

    set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_aV2P_TEST_OPTIONS})
endif()
