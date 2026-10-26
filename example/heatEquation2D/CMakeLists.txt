#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME heatEquation2D)

project(${_TARGET_NAME} LANGUAGES CXX)


################################################################################
# PNGwriter
################################################################################

# find PNGwriter installation
find_package(PNGwriter 0.7.0 CONFIG)

if(PNGwriter_FOUND)
  set(PNGWRITER_ENABLED True)
else()
  set(PNGWRITER_ENABLED False)
endif()

#-------------------------------------------------------------------------------
# Find alpaka.

if(NOT TARGET aV2P::aV2P)
    option(aV2P_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(aV2P_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(aV2P_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/aV2P")
    else()
        find_package(aV2P REQUIRED)
    endif()
endif()

#-------------------------------------------------------------------------------
# Add executable.

aV2P_add_executable(
    ${_TARGET_NAME}
    src/heatEquation2D.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC aV2P::aV2P)
if(PNGwriter_FOUND)
    target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE PNGwriter::PNGwriter)
    target_compile_definitions(${_TARGET_NAME} PRIVATE PNGWRITER_ENABLED)
endif()

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
