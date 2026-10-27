#
# Copyright 2023 Benjamin Worpitz, Axel Hübl, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)

if(NOT TARGET aV2P::aV2P)
    list(APPEND CMAKE_MODULE_PATH "${aV2P_ROOT}")
    find_package(aV2P REQUIRED)
endif()

add_subdirectory(common)

list(APPEND _aV2P_TEST_OPTIONS --colour-mode default)

add_subdirectory(analysis)
add_subdirectory(integ)
add_subdirectory(unit)

# Installed-package regression test: install this build into a scratch prefix and configure the
# consumer with find_package(aV2P CONFIG REQUIRED) without alpaka_DIR.
string(REPLACE ";" "|" _aV2P_PACKAGE_TEST_EXTRA_PREFIX "${CMAKE_PREFIX_PATH}")
add_test(
    NAME packageConsumer
    COMMAND ${CMAKE_COMMAND}
            "-DAV2P_BINARY_DIR=${PROJECT_BINARY_DIR}"
            "-DAV2P_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/aV2P-package-test-prefix"
            "-DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package/consumer"
            "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
            "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
            "-DCMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}"
            "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
            "-DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}"
            "-DEXTRA_PREFIX_PATH=${_aV2P_PACKAGE_TEST_EXTRA_PREFIX}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/RunPackageConsumerTest.cmake")
