#
# Copyright 2023 Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
#

################################################################################
# Required CMake version

cmake_minimum_required(VERSION 3.25)

cmake_policy(SET CMP0091 NEW)

include(CMakePrintHelpers)
include(GNUInstallDirs)

#-------------------------------------------------------------------------------
# Find alpaka version.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/aV2P/version.hpp" aV2P_VERSION_MAJOR_HPP REGEX "#define AV2P_VERSION_MAJOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/aV2P/version.hpp" aV2P_VERSION_MINOR_HPP REGEX "#define AV2P_VERSION_MINOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/aV2P/version.hpp" aV2P_VERSION_PATCH_HPP REGEX "#define AV2P_VERSION_PATCH ")

string(REGEX MATCH "([0-9]+)" aV2P_VERSION_MAJOR  ${aV2P_VERSION_MAJOR_HPP})
string(REGEX MATCH "([0-9]+)" aV2P_VERSION_MINOR  ${aV2P_VERSION_MINOR_HPP})
string(REGEX MATCH "([0-9]+)" aV2P_VERSION_PATCH  ${aV2P_VERSION_PATCH_HPP})

set(PACKAGE_VERSION "${aV2P_VERSION_MAJOR}.${aV2P_VERSION_MINOR}.${aV2P_VERSION_PATCH}")

project(aV2P VERSION      ${aV2P_VERSION_MAJOR}.${aV2P_VERSION_MINOR}.${aV2P_VERSION_PATCH}
               DESCRIPTION  "The alpaka library is a header-only C++20 abstraction library for accelerator development."
               HOMEPAGE_URL "https://github.com/alpaka-group/alpaka"
               LANGUAGES    CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Options and Variants

option(aV2P_BUILD_EXAMPLES "Build the examples" OFF)
option(aV2P_BUILD_BENCHMARKS "Build the benchmarks" OFF)

# Enable the test infrastructure only if alpaka is the top-level project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(aV2P_ENABLE_WERROR "Treat all warnings as errors." OFF)
    option(BUILD_TESTING "Build the testing tree." OFF)
    include(CTest)
endif()

option(aV2P_INSTALL_TEST_HEADER "Install headers of the namespace aV2P::test. Attention, headers are not designed for production code, see documentation." OFF)

include(CMakeDependentOption)

cmake_dependent_option(aV2P_CHECK_HEADERS "Check all alpaka headers as part of the tests whether they can be compiled standalone." OFF BUILD_TESTING OFF)
cmake_dependent_option(aV2P_USE_INTERNAL_CATCH2 "Use internally shipped Catch2" ON "BUILD_TESTING OR aV2P_BUILD_BENCHMARKS" OFF)

# Declare the complete legacy aV2P 2.2 cache variable surface before any
# translation to native alpaka 3 options happens.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/aV2POptions.cmake)

################################################################################
# Internal variables.

# Set found to true initially and set it to false if a required dependency is missing.
set(_aV2P_FOUND TRUE)

# This file's directory.
set(_aV2P_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# Normalize the path (e.g. remove ../)
get_filename_component(_aV2P_ROOT_DIR ${_aV2P_ROOT_DIR} ABSOLUTE)

# Compiler feature tests.
set(_aV2P_FEATURE_TESTS_DIR "${_aV2P_ROOT_DIR}/cmake/tests")

# Add common functions.
set(_aV2P_COMMON_FILE "${_aV2P_ROOT_DIR}/cmake/common.cmake")
include(${_aV2P_COMMON_FILE})

# Add aV2P_ADD_EXECUTABLE function.
set(_aV2P_ADD_EXECUTABLE_FILE "${_aV2P_ROOT_DIR}/cmake/addExecutable.cmake")
include(${_aV2P_ADD_EXECUTABLE_FILE})

# Add aV2P_ADD_LIBRARY function.
set(_aV2P_ADD_LIBRARY_FILE "${_aV2P_ROOT_DIR}/cmake/addLibrary.cmake")
include(${_aV2P_ADD_LIBRARY_FILE})

# Set include directories
set(_aV2P_INCLUDE_DIRECTORY "${_aV2P_ROOT_DIR}/include")
set(_aV2P_SUFFIXED_INCLUDE_DIR "${_aV2P_INCLUDE_DIRECTORY}/aV2P")

# the sequential accelerator is required for the tests and examples
if(aV2P_BUILD_EXAMPLES OR aV2P_BUILD_BENCHMARKS OR BUILD_TESTING)
  if (NOT (aV2P_ACC_GPU_CUDA_ONLY_MODE OR aV2P_ACC_GPU_HIP_ONLY_MODE))
    if (NOT DEFINED aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
      option(aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE "enable alpaka serial accelerator" ON)
    elseif(NOT aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
      set(aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "enable alpaka serial accelerator" FORCE)
    endif()
  else() #CUDA or HIP-only mode
    message(WARNING "CUDA/HIP-only mode enabled: not enabling alpaka serial accelerator (required for examples and tests)")
  endif()
endif()
# Native alpaka 3 owns all backend/dependency configuration. Translate the aV2P
# options first and acquire the pinned native dependency exactly once.
set(aV2P_USE_NATIVE_ALPAKA3 ON)
include(${_aV2P_ROOT_DIR}/cmake/aV2PDependencies.cmake)
include(${_aV2P_ROOT_DIR}/cmake/aV2PCommon.cmake)

# Native alpaka requires every target that links it to be finalized. This file
# defines aV2P_finalize() and the deferred finalize used by the build helpers.
# It is included after the native dependency so the alpaka helpers are available.
include(${_aV2P_ROOT_DIR}/cmake/aV2PFinalize.cmake)

# The native target is linked as a build-interface dependency only. The installed
# aV2PConfig.cmake restores it after find_dependency(alpaka). This keeps the
# exported target free of a non-exported build-tree target.
if(TARGET aV2P AND TARGET alpaka::alpaka)
    target_link_libraries(aV2P INTERFACE "$<BUILD_INTERFACE:alpaka::alpaka>")
endif()

# Add all the source and include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_aV2P_SUFFIXED_INCLUDE_DIR}" "${_aV2P_SUFFIXED_INCLUDE_DIR}" "hpp" _aV2P_FILES_HEADER)
append_recursive_files_add_to_src_group("${_aV2P_SUFFIXED_INCLUDE_DIR}" "${_aV2P_SUFFIXED_INCLUDE_DIR}" "h" _aV2P_FILES_HEADER)

# remove headers of the folder aV2P/test, if aV2P_INSTALL_TEST_HEADER is disabled
if(NOT aV2P_INSTALL_TEST_HEADER)
  list(FILTER _aV2P_FILES_HEADER EXCLUDE REGEX "(.*)/aV2P/test/(.*)")
endif()

append_recursive_files_add_to_src_group("${_aV2P_ROOT_DIR}/script" "${_aV2P_ROOT_DIR}" "sh" _aV2P_FILES_SCRIPT)
set_source_files_properties(${_aV2P_FILES_SCRIPT} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_aV2P_ROOT_DIR}/cmake" "${_aV2P_ROOT_DIR}" "cmake" _aV2P_FILES_CMAKE)
list(APPEND _aV2P_FILES_CMAKE "${_aV2P_ROOT_DIR}/cmake/aV2PConfig.cmake.in" "${_aV2P_ROOT_DIR}/CMakeLists.txt")
set_source_files_properties(${_aV2P_FILES_CMAKE} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_aV2P_ROOT_DIR}/docs/markdown" "${_aV2P_ROOT_DIR}" "md" _aV2P_FILES_DOC)
set_source_files_properties(${_aV2P_FILES_DOC} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_aV2P_ROOT_DIR}/.github" "${_aV2P_ROOT_DIR}" "yml" _aV2P_FILES_OTHER)
list(APPEND _aV2P_FILES_OTHER "${_aV2P_ROOT_DIR}/.clang-format" "${_aV2P_ROOT_DIR}/.gitignore" "${_aV2P_ROOT_DIR}/.zenodo.json" "${_aV2P_ROOT_DIR}/LICENSE" "${_aV2P_ROOT_DIR}/README.md")
set_source_files_properties(${_aV2P_FILES_OTHER} PROPERTIES HEADER_FILE_ONLY TRUE)

if(TARGET aV2P)
    # HACK: Workaround for the limitation that files added to INTERFACE targets (target_sources) can not be marked as PUBLIC or PRIVATE but only as INTERFACE.
    # Therefore those files will be added to projects "linking" to the INTERFACE library, but are not added to the project itself within an IDE.
    add_custom_target("aV2PIde"
                      SOURCES ${_aV2P_FILES_HEADER} ${_aV2P_FILES_SCRIPT} ${_aV2P_FILES_CMAKE} ${_aV2P_FILES_DOC} ${_aV2P_FILES_OTHER})
endif()

################################################################################
# Add subdirectories

# The Intel GPU runtime uses a SIGSEGV handler to migrate shared (managed) memory back to the host.
# Catch2 installs its own POSIX signal handlers at the beginning of each test case, which intercept these SIGSEGVs
# and turn them into fatal errors. Disable the Catch2 POSIX signal handlers when building for Intel GPUs.
if(aV2P_USE_INTERNAL_CATCH2 AND aV2P_ACC_SYCL_ENABLE AND aV2P_SYCL_ONEAPI_GPU)
    message(STATUS "Catch2: disabling POSIX signal handling, as it is not compatible with the Intel GPU runtime")
    set(CATCH_CONFIG_NO_POSIX_SIGNALS ON)
endif()

add_subdirectory(thirdParty)

if(aV2P_BUILD_EXAMPLES)
    add_subdirectory("example/")
endif()

if(aV2P_BUILD_BENCHMARKS)
    add_subdirectory("benchmarks/")
endif()

# Only build the tests if alpaka is the top-level project and BUILD_TESTING is ON
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory("test/")
endif()

################################################################################
# Installation.

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(_aV2P_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/aV2P")

install(TARGETS aV2P
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

write_basic_package_version_file(
    "aV2PConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# The installed package keeps the build-time aV2P option defaults. A consumer
# that sets a legacy or native option explicitly still wins because the
# generated fragment only initialises undefined cache entries.
set(_aV2P_CONFIG_DEFAULTS_FILE "${PROJECT_BINARY_DIR}/aV2PConfigDefaults.cmake")
file(WRITE "${_aV2P_CONFIG_DEFAULTS_FILE}"
     "# Generated by aV2P. Build-time defaults for the legacy option surface.\n")
file(STRINGS "${_aV2P_ROOT_DIR}/cmake/aV2POptionMapping.csv" _aV2P_MAPPING_ROWS)
foreach(_aV2P_ROW IN LISTS _aV2P_MAPPING_ROWS)
    string(STRIP "${_aV2P_ROW}" _aV2P_ROW)
    if(_aV2P_ROW STREQUAL "" OR _aV2P_ROW MATCHES "^#")
        continue()
    endif()
    string(REPLACE "," ";" _aV2P_FIELDS "${_aV2P_ROW}")
    list(GET _aV2P_FIELDS 0 _aV2P_LEGACY)
    list(GET _aV2P_FIELDS 2 _aV2P_TYPE)
    if(_aV2P_LEGACY MATCHES
       "^aV2P_(ACC_|FAST_MATH|FTZ|RELOCATABLE_DEVICE_CODE|CUDA_|HIP_|SYCL_|CXX_STANDARD|DEBUG|BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB|DISABLE_VENDOR_RNG|ASSERT_ACC_ENABLE|USE_MDSPAN)")
        if(DEFINED ${_aV2P_LEGACY})
            set(_aV2P_VALUE "${${_aV2P_LEGACY}}")
            string(REPLACE "\\" "\\\\" _aV2P_VALUE "${_aV2P_VALUE}")
            string(REPLACE "\"" "\\\"" _aV2P_VALUE "${_aV2P_VALUE}")
            file(APPEND "${_aV2P_CONFIG_DEFAULTS_FILE}"
                 "if(NOT DEFINED ${_aV2P_LEGACY})\n  set(${_aV2P_LEGACY} \"${_aV2P_VALUE}\" CACHE ${_aV2P_TYPE} \"aV2P build-time default\")\nendif()\n")
        endif()
    endif()
endforeach()
unset(_aV2P_MAPPING_ROWS)

if(DEFINED aV2P_HAS_STD_ATOMIC_REF AND NOT aV2P_HAS_STD_ATOMIC_REF)
    set(aV2P_CONFIG_NEEDS_BOOST_ATOMIC ON)
else()
    set(aV2P_CONFIG_NEEDS_BOOST_ATOMIC OFF)
endif()

configure_package_config_file(
    "${_aV2P_ROOT_DIR}/cmake/aV2PConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/aV2PConfig.cmake"
    INSTALL_DESTINATION "${_aV2P_INSTALL_CMAKEDIR}")

install(FILES "${PROJECT_BINARY_DIR}/aV2PConfig.cmake"
              "${PROJECT_BINARY_DIR}/aV2PConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/aV2PConfigDefaults.cmake"
        DESTINATION "${_aV2P_INSTALL_CMAKEDIR}")

if(aV2P_INSTALL_TEST_HEADER)
    install(DIRECTORY "${_aV2P_SUFFIXED_INCLUDE_DIR}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
else()
install(DIRECTORY "${_aV2P_SUFFIXED_INCLUDE_DIR}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            PATTERN "test" EXCLUDE)
endif()

install(FILES "${_aV2P_ROOT_DIR}/cmake/addExecutable.cmake"
              "${_aV2P_ROOT_DIR}/cmake/addLibrary.cmake"
              "${_aV2P_ROOT_DIR}/cmake/aV2PCommon.cmake"
              "${_aV2P_ROOT_DIR}/cmake/aV2PFinalize.cmake"
              "${_aV2P_ROOT_DIR}/cmake/aV2POptions.cmake"
              "${_aV2P_ROOT_DIR}/cmake/aV2PNativeOptions.cmake"
              "${_aV2P_ROOT_DIR}/cmake/aV2POptionMapping.csv"
              "${_aV2P_ROOT_DIR}/cmake/common.cmake"
        DESTINATION "${_aV2P_INSTALL_CMAKEDIR}")
install(DIRECTORY "${_aV2P_ROOT_DIR}/cmake/tests"
        DESTINATION "${_aV2P_INSTALL_CMAKEDIR}")

install(TARGETS aV2P EXPORT aV2PTargets)
install(EXPORT aV2PTargets DESTINATION "${_aV2P_INSTALL_CMAKEDIR}" NAMESPACE aV2P::)
