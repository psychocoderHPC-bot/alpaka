#
# SPDX-License-Identifier: MPL-2.0
#
# Acquire the native alpaka 3.0.0 dependency and translate the legacy aV2P
# options exactly once.
#
# Reuse order:
#   1. an already configured alpaka::alpaka target in the same CMake tree
#   2. aV2P_ALPAKA3_SOURCE_DIR (source tree, no network)
#   3. alpaka3_DIR / aV2P_ALPAKA3_DIR / alpaka_DIR (installed package, no network)
#   4. FetchContent with a pinned commit; the populated _deps source is reused
#      on reconfigure and no network access is attempted for it.
#
# FetchContent populates alpaka into CMAKE_BINARY_DIR/_deps/alpaka-src. Set
# aV2P_ALPAKA3_OFFLINE=ON to forbid any network access explicitly.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/aV2PNativeOptions.cmake")

set(aV2P_ALPAKA3_VERSION "3.0.0")
set(aV2P_ALPAKA3_GIT_REPOSITORY "https://github.com/alpaka-group/alpaka.git"
    CACHE STRING "Repository used to fetch native alpaka 3")
set(aV2P_ALPAKA3_GIT_TAG "eb677e8d7f70d79b35df25d9002f94ba669b6f44"
    CACHE STRING "Pinned native alpaka commit (3.0.0)")
set(aV2P_ALPAKA3_SOURCE_DIR ""
    CACHE PATH "Native alpaka 3 source tree to add as a subdirectory (offline)")
set(aV2P_ALPAKA3_DIR ""
    CACHE PATH "Native alpaka 3 package directory containing alpakaConfig.cmake (offline)")
option(aV2P_ALPAKA3_OFFLINE "Forbid network access when fetching native alpaka 3" OFF)

function(_aV2P_alpaka3_version_from_source source_dir out_var)
    set(_version_file "${source_dir}/include/alpaka/version.hpp")
    if(NOT EXISTS "${_version_file}")
        message(FATAL_ERROR
            "aV2P could not find native alpaka 3 sources at '${source_dir}' "
            "(missing include/alpaka/version.hpp).")
    endif()
    file(STRINGS "${_version_file}" _major_line REGEX "#define ALPAKA_VERSION_MAJOR ")
    file(STRINGS "${_version_file}" _minor_line REGEX "#define ALPAKA_VERSION_MINOR ")
    file(STRINGS "${_version_file}" _patch_line REGEX "#define ALPAKA_VERSION_PATCH ")
    string(REGEX MATCH "([0-9]+)" _major "${_major_line}")
    string(REGEX MATCH "([0-9]+)" _minor "${_minor_line}")
    string(REGEX MATCH "([0-9]+)" _patch "${_patch_line}")
    set(${out_var} "${_major}.${_minor}.${_patch}" PARENT_SCOPE)
endfunction()

function(_aV2P_verify_alpaka3_version)
    set(_version "")

    if(DEFINED alpaka_VERSION AND NOT "${alpaka_VERSION}" STREQUAL "")
        set(_version "${alpaka_VERSION}")
    elseif(TARGET alpaka)
        get_target_property(_version alpaka VERSION)
        if(NOT _version)
            set(_version "")
        endif()
    endif()

    if(_version STREQUAL "" AND DEFINED alpaka_SOURCE_DIR AND EXISTS "${alpaka_SOURCE_DIR}")
        _aV2P_alpaka3_version_from_source("${alpaka_SOURCE_DIR}" _version)
    endif()
    if(_version STREQUAL "" AND aV2P_ALPAKA3_SOURCE_DIR)
        _aV2P_alpaka3_version_from_source("${aV2P_ALPAKA3_SOURCE_DIR}" _version)
    endif()

    if(_version STREQUAL "")
        message(FATAL_ERROR
            "aV2P could not determine the native alpaka version. "
            "Set aV2P_ALPAKA3_SOURCE_DIR, aV2P_ALPAKA3_DIR, alpaka3_DIR or alpaka_DIR.")
    endif()

    if(NOT _version VERSION_EQUAL "${aV2P_ALPAKA3_VERSION}")
        message(FATAL_ERROR
            "aV2P requires native alpaka ${aV2P_ALPAKA3_VERSION} but found ${_version}. "
            "Set aV2P_ALPAKA3_SOURCE_DIR or alpaka_DIR to an alpaka ${aV2P_ALPAKA3_VERSION} tree.")
    endif()

    message(STATUS "aV2P: using native alpaka ${_version} (pinned ${aV2P_ALPAKA3_VERSION})")
endfunction()

if(TARGET alpaka::alpaka)
    message(STATUS "aV2P: reusing existing target alpaka::alpaka from the parent project")
elseif(aV2P_ALPAKA3_SOURCE_DIR)
    _aV2P_alpaka3_version_from_source("${aV2P_ALPAKA3_SOURCE_DIR}" _aV2P_SOURCE_VERSION)
    if(NOT _aV2P_SOURCE_VERSION VERSION_EQUAL "${aV2P_ALPAKA3_VERSION}")
        message(FATAL_ERROR
            "aV2P_ALPAKA3_SOURCE_DIR points to alpaka ${_aV2P_SOURCE_VERSION} but "
            "${aV2P_ALPAKA3_VERSION} is required.")
    endif()
    message(STATUS "aV2P: adding native alpaka ${_aV2P_SOURCE_VERSION} from ${aV2P_ALPAKA3_SOURCE_DIR}")
    set(_aV2P_SKIP_INSTALL_SAVED "${CMAKE_SKIP_INSTALL_RULES}")
    set(CMAKE_SKIP_INSTALL_RULES ON)
    add_subdirectory("${aV2P_ALPAKA3_SOURCE_DIR}"
                     "${CMAKE_BINARY_DIR}/_aV2P-alpaka3-build"
                     EXCLUDE_FROM_ALL)
    set(CMAKE_SKIP_INSTALL_RULES "${_aV2P_SKIP_INSTALL_SAVED}")
    unset(_aV2P_SKIP_INSTALL_SAVED)
elseif(aV2P_ALPAKA3_DIR OR alpaka3_DIR OR alpaka_DIR)
    if(aV2P_ALPAKA3_DIR AND NOT alpaka_DIR)
        set(alpaka_DIR "${aV2P_ALPAKA3_DIR}" CACHE PATH
            "Native alpaka 3 package directory (set by aV2P_ALPAKA3_DIR)")
    elseif(alpaka3_DIR AND NOT alpaka_DIR)
        set(alpaka_DIR "${alpaka3_DIR}" CACHE PATH
            "Native alpaka 3 package directory (set from alpaka3_DIR)")
    endif()
    message(STATUS "aV2P: finding native alpaka ${aV2P_ALPAKA3_VERSION} package in ${alpaka_DIR}")
    find_package(alpaka ${aV2P_ALPAKA3_VERSION} CONFIG REQUIRED)
else()
    include(FetchContent)

    set(_aV2P_FETCHED_SOURCE "${FETCHCONTENT_BASE_DIR}/alpaka-src")
    if(NOT aV2P_ALPAKA3_OFFLINE AND EXISTS "${_aV2P_FETCHED_SOURCE}/CMakeLists.txt")
        message(STATUS "aV2P: reusing already fetched native alpaka sources in ${_aV2P_FETCHED_SOURCE}")
        set(aV2P_ALPAKA3_OFFLINE ON)
    endif()

    if(aV2P_ALPAKA3_OFFLINE)
        if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_ALPAKA
           AND NOT EXISTS "${_aV2P_FETCHED_SOURCE}/CMakeLists.txt")
            message(FATAL_ERROR
                "aV2P_ALPAKA3_OFFLINE is ON but no populated alpaka source tree exists at "
                "'${_aV2P_FETCHED_SOURCE}' and FETCHCONTENT_SOURCE_DIR_ALPAKA is not set. "
                "Set aV2P_ALPAKA3_SOURCE_DIR/alpaka_DIR or pre-populate the source tree.")
        endif()
        set(FETCHCONTENT_FULLY_DISCONNECTED ON)
        set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    endif()

    FetchContent_Declare(alpaka
        GIT_REPOSITORY "${aV2P_ALPAKA3_GIT_REPOSITORY}"
        GIT_TAG "${aV2P_ALPAKA3_GIT_TAG}"
        GIT_SHALLOW FALSE)

    # Native alpaka must not add its own install rules to the aV2P install.
    # Otherwise installing aV2P would also write alpaka headers/config and could
    # clobber an existing alpaka installation.
    set(_aV2P_SKIP_INSTALL_SAVED "${CMAKE_SKIP_INSTALL_RULES}")
    set(CMAKE_SKIP_INSTALL_RULES ON)
    FetchContent_MakeAvailable(alpaka)
    set(CMAKE_SKIP_INSTALL_RULES "${_aV2P_SKIP_INSTALL_SAVED}")
    unset(_aV2P_SKIP_INSTALL_SAVED)
endif()

_aV2P_verify_alpaka3_version()
