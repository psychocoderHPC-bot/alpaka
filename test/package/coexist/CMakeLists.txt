#
# SPDX-License-Identifier: MPL-2.0
#
# Coexistence consumer: installed aV2P and installed native alpaka 3 in the same
# translation unit. alpaka_finalize is required by native alpaka 3 whenever one
# of its targets is linked.
#
cmake_minimum_required(VERSION 3.25)
project(aV2PCoexistConsumer LANGUAGES CXX)

find_package(aV2P CONFIG REQUIRED)
find_package(alpaka 3.0.0 CONFIG REQUIRED)

add_executable(aV2P-coexist main.cpp)
target_link_libraries(aV2P-coexist PRIVATE aV2P::aV2P alpaka::alpaka)
alpaka_finalize(aV2P-coexist)
