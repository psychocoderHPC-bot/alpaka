#
# Copyright 2026 Alexander Matthes, Axel Huebl, Benjamin Worpitz
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME aV2PInterop)

project(${_TARGET_NAME} LANGUAGES CXX)


#-------------------------------------------------------------------------------
# Find aV2P.

if(NOT TARGET aV2P::aV2P)
    option(aV2P_USE_SOURCE_TREE "Use aV2P's source tree instead of an aV2P installation" OFF)

    if(aV2P_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(aV2P_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/aV2P")
    else()
        find_package(aV2P REQUIRED)
    endif()
endif()

#-------------------------------------------------------------------------------
# Native alpaka 3 dependency.
#
# The example includes both API headers in one translation unit, so the native
# target has to be available in both the source-tree and the installed case.
# aV2P acquires it as its own dependency; an already defined target wins and
# only a standalone consumer has to find the package itself.

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka CONFIG REQUIRED)
endif()

#-------------------------------------------------------------------------------
# Add executable.

aV2P_add_executable(
    ${_TARGET_NAME}
    src/aV2PInterop.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC aV2P::aV2P
           alpaka::alpaka)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
