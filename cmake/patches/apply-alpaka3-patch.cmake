#
# SPDX-License-Identifier: MPL-2.0
#
# Apply a patch to a native alpaka 3 source tree exactly once.
#
# Invoked from FetchContent (PATCH_COMMAND) and from aV2PDependencies.cmake for an
# aV2P_ALPAKA3_SOURCE_DIR tree.  The reverse check makes a second invocation a no-op,
# so reconfiguring a build tree never fails with "patch does not apply".
#
# Required variables:
#   SOURCE_DIR  - native alpaka source tree
#   PATCH_FILE  - patch file, applied with -p1 relative to SOURCE_DIR

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE)
    message(FATAL_ERROR "apply-alpaka3-patch.cmake requires SOURCE_DIR and PATCH_FILE")
endif()
if(NOT EXISTS "${SOURCE_DIR}")
    message(FATAL_ERROR "aV2P: native alpaka source tree does not exist: ${SOURCE_DIR}")
endif()
if(NOT EXISTS "${PATCH_FILE}")
    message(FATAL_ERROR "aV2P: native alpaka patch does not exist: ${PATCH_FILE}")
endif()

execute_process(
    COMMAND git apply --reverse --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _aV2P_patch_already
    OUTPUT_QUIET
    ERROR_QUIET)
if(_aV2P_patch_already EQUAL 0)
    message(STATUS "aV2P: native alpaka patch already applied: ${PATCH_FILE}")
    return()
endif()

execute_process(
    COMMAND git apply --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _aV2P_patch_check
    OUTPUT_VARIABLE _aV2P_patch_check_out
    ERROR_VARIABLE _aV2P_patch_check_err)
if(NOT _aV2P_patch_check EQUAL 0)
    message(FATAL_ERROR
        "aV2P: cannot apply the native alpaka patch '${PATCH_FILE}' to '${SOURCE_DIR}'. "
        "The tree is not the pinned alpaka 3.0.0 revision or was modified.\n"
        "${_aV2P_patch_check_out}${_aV2P_patch_check_err}")
endif()

execute_process(
    COMMAND git apply "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _aV2P_patch_apply
    OUTPUT_VARIABLE _aV2P_patch_apply_out
    ERROR_VARIABLE _aV2P_patch_apply_err)
if(NOT _aV2P_patch_apply EQUAL 0)
    message(FATAL_ERROR
        "aV2P: failed to apply the native alpaka patch '${PATCH_FILE}' to '${SOURCE_DIR}'.\n"
        "${_aV2P_patch_apply_out}${_aV2P_patch_apply_err}")
endif()
message(STATUS "aV2P: applied native alpaka patch: ${PATCH_FILE}")
