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
