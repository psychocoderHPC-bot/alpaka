#
# SPDX-License-Identifier: MPL-2.0
#
# Translate the legacy aV2P 2.2 CMake options to native alpaka 3.0.0 options.
#
# This is the only place where aV2P writes alpaka_* cache variables. It runs
# before alpaka is found / fetched / added, so a single native configuration is
# produced. If the application already set a native alpaka_* variable to a
# different value a hard error names both options instead of overwriting it.
#
# The table is kept in aV2POptionMapping.csv so tooling and the migration guide
# can consume it.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/aV2POptions.cmake")

function(_aV2P_native_conflict legacy native requested actual)
    message(FATAL_ERROR
        "Conflicting backend configuration: aV2P option '${legacy}' maps to native alpaka option "
        "'${native}'='${requested}', but '${native}' is already set to '${actual}'. "
        "Set only one of '${legacy}' and '${native}'.")
endfunction()

function(_aV2P_set_native_option legacy native type requested)
    # Remember the value aV2P wrote so that a later reconfigure may update its
    # own mapped value. A native value without this marker is owned by the
    # application and must never be overwritten.
    set(_marker "_aV2P_NATIVE_MAPPED_${native}")

    if(DEFINED ${native})
        if(type STREQUAL "BOOL")
            set(_same FALSE)
            if(requested AND ${${native}})
                set(_same TRUE)
            elseif(NOT requested AND NOT ${${native}})
                set(_same TRUE)
            endif()
        else()
            if("${${native}}" STREQUAL "${requested}")
                set(_same TRUE)
            else()
                set(_same FALSE)
            endif()
        endif()

        if(NOT _same)
            set(_owned FALSE)
            if(DEFINED ${_marker} AND "${${_marker}}" STREQUAL "${${native}}")
                set(_owned TRUE)
            endif()
            if(NOT _owned)
                _aV2P_native_conflict("${legacy}" "${native}" "${requested}" "${${native}}")
            endif()
            set(${native} "${requested}" CACHE ${type}
                "Mapped from aV2P option ${legacy}" FORCE)
        endif()
    else()
        set(${native} "${requested}" CACHE ${type} "Mapped from aV2P option ${legacy}")
    endif()

    set(${_marker} "${${native}}" CACHE INTERNAL "aV2P mapped value for ${native}" FORCE)
endfunction()

function(_aV2P_apply_native_option_mapping)
    set(_mapping_file "${CMAKE_CURRENT_LIST_DIR}/aV2POptionMapping.csv")
    if(NOT EXISTS "${_mapping_file}")
        message(FATAL_ERROR "aV2P option mapping table not found: ${_mapping_file}")
    endif()

    file(STRINGS "${_mapping_file}" _rows)
    foreach(_row IN LISTS _rows)
        string(STRIP "${_row}" _row)
        if(_row STREQUAL "" OR _row MATCHES "^#")
            continue()
        endif()

        string(REPLACE "," ";" _fields "${_row}")
        list(LENGTH _fields _nfields)
        if(NOT _nfields EQUAL 6)
            message(FATAL_ERROR "Malformed aV2P option mapping row: ${_row}")
        endif()
        list(GET _fields 0 _legacy)
        list(GET _fields 1 _native)
        list(GET _fields 2 _type)
        list(GET _fields 3 _mode)
        list(GET _fields 4 _default)
        list(GET _fields 5 _note)

        set(_value "${${_legacy}}")

        if(_mode STREQUAL "local")
            # aV2P-local only. No native option is touched.
            continue()
        elseif(_mode STREQUAL "on")
            if(_value)
                _aV2P_set_native_option("${_legacy}" "${_native}" "${_type}" ON)
            endif()
        elseif(_mode STREQUAL "forward")
            if(NOT "${_value}" STREQUAL "${_default}")
                # The application explicitly changed the legacy option.
                _aV2P_set_native_option("${_legacy}" "${_native}" "${_type}" "${_value}")
            elseif(NOT DEFINED ${_native})
                # Keep the 2.2 default, but never replace a native value.
                _aV2P_set_native_option("${_legacy}" "${_native}" "${_type}" "${_value}")
            endif()
        elseif(_mode STREQUAL "unsupported")
            if(NOT "${_value}" STREQUAL "${_default}")
                message(FATAL_ERROR
                    "aV2P option '${_legacy}' is set to '${_value}', but native alpaka 3.0.0 "
                    "cannot provide it: ${_note}. Set '${_legacy}' to '${_default}' or select a "
                    "supported alpaka 3 backend.")
            endif()
        elseif(_mode STREQUAL "cuda_unsupported")
            if(aV2P_ACC_GPU_CUDA_ENABLE AND NOT "${_value}" STREQUAL "${_default}")
                message(FATAL_ERROR
                    "aV2P option '${_legacy}' is set to '${_value}', but native alpaka 3.0.0 "
                    "cannot provide it: ${_note}. Set '${_legacy}' to '${_default}'.")
            endif()
        elseif(_mode STREQUAL "sycl_device")
            # A oneAPI device kind selected through a legacy option has to be active on the native
            # side, and a kind that is not selected must not be compiled. Forward in both directions
            # only while the SYCL backend is enabled; otherwise the native option stays untouched so
            # a plain alpaka 3 (non-aV2P) configuration keeps its default.
            if(aV2P_ACC_SYCL_ENABLE)
                if(_value)
                    _aV2P_set_native_option("${_legacy}" "${_native}" "${_type}" ON)
                else()
                    _aV2P_set_native_option("${_legacy}" "${_native}" "${_type}" OFF)
                endif()
            endif()
        elseif(_mode STREQUAL "sycl_unsupported")
            if(aV2P_ACC_SYCL_ENABLE AND NOT "${_value}" STREQUAL "${_default}")
                message(FATAL_ERROR
                    "aV2P option '${_legacy}' is set to '${_value}', but native alpaka 3.0.0 "
                    "cannot provide it: ${_note}. Set '${_legacy}' to '${_default}' or select a "
                    "supported alpaka 3 backend.")
            endif()
        elseif(_mode STREQUAL "sycl_warn")
            if(aV2P_ACC_SYCL_ENABLE AND NOT "${_value}" STREQUAL "${_default}")
                message(WARNING
                    "aV2P option '${_legacy}' is set to '${_value}', but native alpaka 3.0.0 "
                    "uses a different mechanism: ${_note}. The aV2P value is not forwarded.")
            endif()
        else()
            message(FATAL_ERROR "Unknown aV2P option mapping mode '${_mode}' for '${_legacy}'.")
        endif()
    endforeach()
endfunction()

_aV2P_apply_native_option_mapping()

# The SYCL backend selects exactly one aV2P device kind per accelerator tag. Enabling it without any
# oneAPI target would compile native SYCL kernels for none of the device kinds aV2P exposes, leaving
# the build with no usable aV2P SYCL accelerator. Fail with a clear selection error instead.
if(aV2P_ACC_SYCL_ENABLE
   AND NOT (aV2P_SYCL_ONEAPI_CPU
            OR aV2P_SYCL_ONEAPI_FPGA
            OR aV2P_SYCL_ONEAPI_GPU
            OR aV2P_SYCL_ONEAPI_GPU_NVIDIA
            OR aV2P_SYCL_ONEAPI_GPU_AMD))
    message(FATAL_ERROR
        "aV2P_ACC_SYCL_ENABLE=ON but no oneAPI target is selected. Set at least one of "
        "aV2P_SYCL_ONEAPI_CPU, aV2P_SYCL_ONEAPI_FPGA, aV2P_SYCL_ONEAPI_GPU, "
        "aV2P_SYCL_ONEAPI_GPU_NVIDIA, aV2P_SYCL_ONEAPI_GPU_AMD to ON.")
endif()

# ONLY_MODE has no native switch in alpaka 3: native alpaka already permits
# mixing with vendor code. Keep the 2.2 validation and explain the translation.
if(aV2P_ACC_GPU_CUDA_ONLY_MODE)
    message(STATUS
        "aV2P_ACC_GPU_CUDA_ONLY_MODE=ON: kept as a 2.2 compatibility validation. "
        "alpaka 3 has no global only-mode switch and allows mixing with native CUDA.")
endif()
if(aV2P_ACC_GPU_HIP_ONLY_MODE)
    message(STATUS
        "aV2P_ACC_GPU_HIP_ONLY_MODE=ON: kept as a 2.2 compatibility validation. "
        "alpaka 3 has no global only-mode switch and allows mixing with native HIP.")
endif()
