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
