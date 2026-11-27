#
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)

project(alpaka-spack-consumer LANGUAGES CXX)

set(ALPAKA_TEST_DEVICE "host" CACHE STRING "Backend used by the consumer: host, cuda, hip, or oneapi")

find_package(alpaka CONFIG REQUIRED)

add_executable(alpaka_spack_consumer main.cpp)
target_link_libraries(alpaka_spack_consumer PRIVATE alpaka::alpaka)
alpaka_finalize(alpaka_spack_consumer)

if(ALPAKA_TEST_DEVICE STREQUAL "cuda")
    enable_language(CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(alpaka_spack_consumer PRIVATE ALPAKA_TEST_CUDA)
elseif(ALPAKA_TEST_DEVICE STREQUAL "hip")
    enable_language(HIP)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_compile_definitions(alpaka_spack_consumer PRIVATE ALPAKA_TEST_HIP)
elseif(ALPAKA_TEST_DEVICE STREQUAL "oneapi")
    target_compile_definitions(alpaka_spack_consumer PRIVATE ALPAKA_TEST_ONEAPI)
endif()

enable_testing()
add_test(NAME alpaka_spack_consumer COMMAND alpaka_spack_consumer)
