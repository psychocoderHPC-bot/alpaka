# Copyright 2023 Bernhard Manfred Gruber
# SPDX-License-Identifier: ISC

cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(aV2P-ls LANGUAGES CXX)
if(NOT TARGET aV2P::aV2P)
    option(aV2P_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(aV2P_USE_SOURCE_TREE)
        set(aV2P_BUILD_EXAMPLES OFF) # Don't build the examples recursively
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/aV2P")
    else()
        find_package(aV2P REQUIRED)
    endif()
endif()

aV2P_add_executable(${PROJECT_NAME} src/ls.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC aV2P::aV2P)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER example)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
