#
# SPDX-License-Identifier: MPL-2.0
#
# Installed-package regression test.
#
# Installs the current aV2P build tree into a scratch prefix, then configures and builds
# test/package/consumer with find_package(aV2P CONFIG REQUIRED).  Only the scratch prefix is on
# CMAKE_PREFIX_PATH; alpaka_DIR is deliberately not passed.  The consumer therefore proves that the
# install is self-contained and that aV2PConfig.cmake selects the bundled native alpaka package.

foreach(_var AV2P_BINARY_DIR AV2P_INSTALL_PREFIX CONSUMER_SOURCE_DIR CMAKE_GENERATOR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "RunPackageConsumerTest.cmake requires ${_var}")
    endif()
endforeach()

file(REMOVE_RECURSE "${AV2P_INSTALL_PREFIX}" "${AV2P_BINARY_DIR}/package-consumer-build")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${AV2P_BINARY_DIR}" --prefix "${AV2P_INSTALL_PREFIX}"
    RESULT_VARIABLE _install_result
    OUTPUT_VARIABLE _install_out
    ERROR_VARIABLE _install_err)
if(NOT _install_result EQUAL 0)
    message(FATAL_ERROR "aV2P: package test install failed:\n${_install_out}${_install_err}")
endif()

if(NOT EXISTS "${AV2P_INSTALL_PREFIX}/include/aV2P/alpaka.hpp")
    message(FATAL_ERROR "aV2P: the install prefix has no include/aV2P/alpaka.hpp")
endif()
if(NOT EXISTS "${AV2P_INSTALL_PREFIX}/include/alpaka/alpaka.hpp")
    message(FATAL_ERROR "aV2P: the install prefix has no bundled include/alpaka/alpaka.hpp")
endif()
if(NOT EXISTS "${AV2P_INSTALL_PREFIX}/lib/cmake/alpaka/alpakaConfig.cmake"
   AND NOT EXISTS "${AV2P_INSTALL_PREFIX}/lib64/cmake/alpaka/alpakaConfig.cmake")
    message(FATAL_ERROR "aV2P: the install prefix has no bundled alpaka CMake package")
endif()

set(_prefix_path "${AV2P_INSTALL_PREFIX}")
if(DEFINED EXTRA_PREFIX_PATH AND NOT "${EXTRA_PREFIX_PATH}" STREQUAL "")
    string(REPLACE "|" ";" _extra "${EXTRA_PREFIX_PATH}")
    list(APPEND _prefix_path ${_extra})
endif()

set(_configure_command
    "${CMAKE_COMMAND}"
    -S "${CONSUMER_SOURCE_DIR}"
    -B "${AV2P_BINARY_DIR}/package-consumer-build"
    -G "${CMAKE_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${_prefix_path}"
    "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
    "-DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON")
if(DEFINED CMAKE_BUILD_TYPE AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
    list(APPEND _configure_command "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()
foreach(_compiler CMAKE_CXX_COMPILER CMAKE_CUDA_COMPILER CMAKE_HIP_COMPILER
                  CMAKE_CUDA_ARCHITECTURES CMAKE_HIP_ARCHITECTURES)
    if(DEFINED ${_compiler} AND NOT "${${_compiler}}" STREQUAL "")
        list(APPEND _configure_command "-D${_compiler}=${${_compiler}}")
    endif()
endforeach()

execute_process(
    COMMAND ${_configure_command}
    RESULT_VARIABLE _configure_result
    OUTPUT_VARIABLE _configure_out
    ERROR_VARIABLE _configure_err)
if(NOT _configure_result EQUAL 0)
    message(FATAL_ERROR "aV2P: package consumer configuration failed:\n${_configure_out}${_configure_err}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${AV2P_BINARY_DIR}/package-consumer-build"
    RESULT_VARIABLE _build_result
    OUTPUT_VARIABLE _build_out
    ERROR_VARIABLE _build_err)
if(NOT _build_result EQUAL 0)
    message(FATAL_ERROR "aV2P: package consumer build failed:\n${_build_out}${_build_err}")
endif()

execute_process(
    COMMAND "${AV2P_BINARY_DIR}/package-consumer-build/aV2P-consumer"
    RESULT_VARIABLE _run_result
    OUTPUT_VARIABLE _run_out
    ERROR_VARIABLE _run_err)
if(NOT _run_result EQUAL 0)
    message(FATAL_ERROR "aV2P: package consumer failed to run:\n${_run_out}${_run_err}")
endif()

message(STATUS "aV2P: package consumer OK: ${_run_out}")
