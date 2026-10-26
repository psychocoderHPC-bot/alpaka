#
# SPDX-License-Identifier: MPL-2.0
#
# Installed-package consumer: find_package(aV2P CONFIG REQUIRED) and use the
# renamed header plus the aV2P::aV2P target. No GPU and no native alpaka header
# is needed here.
#
cmake_minimum_required(VERSION 3.25)
project(aV2PPackageConsumer LANGUAGES CXX)

find_package(aV2P CONFIG REQUIRED)

add_executable(aV2P-consumer main.cpp)
target_link_libraries(aV2P-consumer PRIVATE aV2P::aV2P)
