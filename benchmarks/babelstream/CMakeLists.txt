#
# Copyright 2023 Erik Zenker, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(babelstream LANGUAGES CXX)

if(NOT TARGET aV2P::aV2P)
    option(aV2P_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(aV2P_USE_SOURCE_TREE)
        # Don't build the benchmarks recursively
        set(aV2P_BUILD_BENCHMARKS OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/aV2P")
    else()
        find_package(aV2P REQUIRED)
    endif()
endif()


set(_TARGET_NAME "babelstream")
append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

aV2P_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})

target_include_directories(
 ${_TARGET_NAME}
 PRIVATE "src")

target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER benchmarks/babelstream)

#Run as a ctest
if(aV2P_CI)
    # Only run for release builds since this is a benchmark
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
       add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
    endif()
else()
    # For a normal benchmark test, number of samples should be equal to the default value.
    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
endif()
