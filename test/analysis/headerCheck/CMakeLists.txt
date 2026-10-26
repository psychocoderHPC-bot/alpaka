#
# Copyright 2021 Benjamin Worpitz, Axel Huebl, Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
#

if(aV2P_CHECK_HEADERS)

    set(_TARGET_NAME "headerCheckTest")

    #---------------------------------------------------------------------------
    # Create source files.
    set(_aV2P_INCLUDE_DIRECTORY "${_aV2P_ROOT_DIR}/include")
    set(_aV2P_SUFFIXED_INCLUDE_DIR "${_aV2P_INCLUDE_DIRECTORY}/aV2P")
    append_recursive_files("${_aV2P_SUFFIXED_INCLUDE_DIR}" "hpp" "_aV2P_FILES_HEADER")

    set(_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

    file(REMOVE_RECURSE ${_GENERATED_SOURCE_DIR})

    foreach(_HEADER_FILE ${_aV2P_FILES_HEADER})
        # Remove the parent directory from the path.
        # NOTE: This is not correct because it does not only replace at the beginning of the string.
        #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
        string(
            REPLACE "${_aV2P_SUFFIXED_INCLUDE_DIR}/" ""
            _HEADER_FILE
            "${_HEADER_FILE}")
        set(_SOURCE_FILE "${_GENERATED_SOURCE_DIR}/${_HEADER_FILE}.cpp")
        file(WRITE "${_SOURCE_FILE}" "#include <aV2P/${_HEADER_FILE}>\n#include <aV2P/${_HEADER_FILE}>\n")
    endforeach()

    #---------------------------------------------------------------------------
    # Add executable.

    append_recursive_files_add_to_src_group("${_GENERATED_SOURCE_DIR}" "${_GENERATED_SOURCE_DIR}" "cpp" "_FILES_SOURCE")
    list(APPEND _FILES_SOURCE "src/main.cpp")

    # Always add all files to the target executable build call to add them to the build project.
    aV2P_add_executable(
        ${_TARGET_NAME}
        ${_FILES_SOURCE})
    target_link_libraries(
        ${_TARGET_NAME}
        PRIVATE common)

    set_target_properties(headerCheckTest PROPERTIES FOLDER "test/analysis")

    add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_aV2P_TEST_OPTIONS})

endif()
