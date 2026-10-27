#
# Copyright 2022 Andrea Bocci
# SPDX-License-Identifier: MPL-2.0
#

set(_TARGET_NAME "hostOnlyAPITest")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

target_compile_definitions(${_TARGET_NAME} PRIVATE AV2P_HOST_ONLY)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

if(aV2P_ACC_GPU_CUDA_ENABLE OR aV2P_ACC_GPU_HIP_ENABLE)
    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_aV2P_TEST_OPTIONS})
elseif(NOT aV2P_ACC_SYCL_ENABLE)
    # Host-only configuration: the test exercises the host API and the enabled host accelerators and
    # has no GPU dependency, so it must actually run instead of being built but never registered.
    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_aV2P_TEST_OPTIONS})
endif()

# Regression test for the native alpaka 3 launch guard: the same host-only TU that constructs a
# device and queue above must fail to compile a kernel enqueue with one clear static_assert.
if((aV2P_ACC_GPU_CUDA_ENABLE OR aV2P_ACC_GPU_HIP_ENABLE) AND aV2P_ALPAKA3_INCLUDE_DIR)
    set(_aV2P_NEGATIVE_FLAGS
        "-DAV2P_HOST_ONLY"
        "-I${_aV2P_INCLUDE_DIRECTORY}"
        "-I${aV2P_ALPAKA3_INCLUDE_DIR}")
    if(aV2P_ACC_GPU_CUDA_ENABLE)
        find_package(CUDAToolkit REQUIRED)
        list(APPEND _aV2P_NEGATIVE_FLAGS "-DAV2P_ACC_GPU_CUDA_ENABLED")
        foreach(_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
            list(APPEND _aV2P_NEGATIVE_FLAGS "-I${_dir}")
        endforeach()
    endif()
    if(aV2P_ACC_GPU_HIP_ENABLE)
        find_package(hip REQUIRED)
        list(APPEND _aV2P_NEGATIVE_FLAGS "-DAV2P_ACC_GPU_HIP_ENABLED")
        foreach(_dir IN LISTS hip_INCLUDE_DIRS)
            list(APPEND _aV2P_NEGATIVE_FLAGS "-I${_dir}")
        endforeach()
    endif()

    configure_file(
        "negative/expect-compile-failure.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/expect-compile-failure.cmake"
        @ONLY)
    add_test(
        NAME hostOnlyKernelLaunchStaticAssert
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/expect-compile-failure.cmake")
endif()
