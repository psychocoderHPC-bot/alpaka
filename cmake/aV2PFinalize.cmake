#
# SPDX-License-Identifier: MPL-2.0
#
# aV2P finalize integration.
#
# aV2P 3.x adapters include native alpaka 3 headers. Native alpaka requires every
# target that links one of its targets to be finalized with alpaka_finalize()
# before any of its headers are compiled. The native guards in
# alpaka/core/config.hpp turn an unfinalized target into a hard compile error.
#
# aV2P provides two mechanisms so that a 2.2-style consumer does not need a new
# CMake step:
#
#   1. aV2P_add_executable()/aV2P_add_library() schedule a finalize call for the
#      target. The call runs at the end of the directory in which the target was
#      created, i.e. after the usual target_link_libraries(<target> ... aV2P::aV2P)
#      call and while source-file properties and relative source paths still
#      resolve in the target's own directory scope.
#   2. aV2P_finalize(<target>) is the documented equivalent of alpaka_finalize()
#      for targets created with plain add_executable()/add_library().
#
# The native ALPAKA_CMAKE_TARGET_*_FINALIZE_CALLED definitions are intentionally
# not injected through the aV2P interface target: finalize also sets source
# language properties and, for GPU backends, separable-compilation and language
# standard properties that cannot be propagated by an INTERFACE target.

include_guard(GLOBAL)

#-------------------------------------------------------------------------------
# Native alpaka detection.

function(_aV2P_has_native_alpaka out_var)
    if(COMMAND alpaka_internal_finalize OR COMMAND alpaka_finalize)
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

#-------------------------------------------------------------------------------
# Public entry point.

function(aV2P_finalize target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR
            "aV2P_finalize(): '${target}' is not a CMake target. Call "
            "aV2P_finalize() after add_executable()/add_library() and after "
            "target_link_libraries(${target} ... aV2P::aV2P).")
    endif()

    # A target must only be finalized once. Track aV2P's own bookkeeping and also
    # accept targets that were finalized through native alpaka_finalize().
    get_target_property(_aV2P_done "${target}" aV2P_FINALIZE_DONE)
    if(_aV2P_done)
        return()
    endif()
    get_target_property(_aV2P_defs "${target}" COMPILE_DEFINITIONS)
    if(_aV2P_defs AND "${_aV2P_defs}" MATCHES "ALPAKA_CMAKE_TARGET_[A-Z]+_FINALIZE_CALLED")
        set_property(TARGET "${target}" PROPERTY aV2P_FINALIZE_DONE TRUE)
        return()
    endif()

    # Native alpaka can reference toolkit imported targets (for example
    # CUDA::cuda_driver) that were created in a sibling or child directory. Such
    # imported targets are not visible in this directory scope, so acquire the
    # toolkit packages here as well before the native finalize call runs.
    if(aV2P_ACC_GPU_CUDA_ENABLE)
        find_package(CUDAToolkit REQUIRED)
    endif()
    if(aV2P_ACC_GPU_HIP_ENABLE)
        find_package(hip REQUIRED)
    endif()

    if(COMMAND alpaka_get_targets)
        set(_aV2P_alpaka_targets "")
        alpaka_get_targets("${target}" _aV2P_alpaka_targets)
        if(NOT _aV2P_alpaka_targets)
            message(FATAL_ERROR
                "aV2P_finalize(): target '${target}' does not link aV2P or native "
                "alpaka. Link 'aV2P::aV2P' (or an interface target that links it) "
                "before calling aV2P_finalize().")
        endif()
    endif()

    if(COMMAND alpaka_internal_finalize)
        # The build tree keeps the native link behind a BUILD_INTERFACE generator
        # expression so the installed export stays free of build-tree targets.
        # alpaka_get_targets() cannot look through that expression, so materialize
        # the real target for the duration of the finalize call. This makes the
        # host/headers and selected device backends visible to alpaka, which is
        # required for source-language and separable-compilation properties.
        set(_aV2P_patched FALSE)
        if(TARGET aV2P)
            get_target_property(_aV2P_iface aV2P INTERFACE_LINK_LIBRARIES)
            if(_aV2P_iface AND "${_aV2P_iface}" MATCHES "BUILD_INTERFACE:alpaka::alpaka")
                string(REPLACE "$<BUILD_INTERFACE:alpaka::alpaka>" "alpaka" _aV2P_iface_new "${_aV2P_iface}")
                set_target_properties(aV2P PROPERTIES INTERFACE_LINK_LIBRARIES "${_aV2P_iface_new}")
                set(_aV2P_patched TRUE)
            endif()
        endif()
        alpaka_internal_finalize("${target}")
        if(_aV2P_patched)
            set_target_properties(aV2P PROPERTIES INTERFACE_LINK_LIBRARIES "${_aV2P_iface}")
            unset(_aV2P_iface)
        endif()
    elseif(COMMAND alpaka_finalize)
        alpaka_finalize("${target}")
    else()
        message(FATAL_ERROR
            "aV2P_finalize(): the native alpaka CMake helpers are not available. "
            "aV2P requires native alpaka 3 to be found via find_package(alpaka) "
            "or added as a subdirectory before aV2P_finalize() is called.")
    endif()

    set_property(TARGET "${target}" PROPERTY aV2P_FINALIZE_DONE TRUE)
endfunction()

#-------------------------------------------------------------------------------
# Deferred processing: finalize targets registered by the aV2P build helpers,
# auto-finalize targets that link aV2P through a project interface target, and
# report targets that link aV2P directly without calling aV2P_finalize().

function(_aV2P_schedule_finalize target)
    # Keep the bookkeeping directory-local. A deferred call without DIRECTORY
    # runs at the end of the directory scope in which it was registered, which is
    # exactly the directory that created the target. Deferring to a global
    # directory would move source-property and relative-path changes out of the
    # target's own scope.
    set_property(DIRECTORY APPEND PROPERTY _aV2P_FINALIZE_TARGETS "${target}")
    get_property(_aV2P_defer_scheduled DIRECTORY PROPERTY _aV2P_DEFER_SCHEDULED)
    if(NOT _aV2P_defer_scheduled)
        set_property(DIRECTORY PROPERTY _aV2P_DEFER_SCHEDULED TRUE)
        cmake_language(DEFER CALL _aV2P_process_directory_targets "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endfunction()

function(_aV2P_collect_buildsystem_targets directory out_var)
    set(_aV2P_targets "")
    get_property(_aV2P_here DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND _aV2P_targets ${_aV2P_here})
    get_property(_aV2P_subdirs DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(_aV2P_subdir IN LISTS _aV2P_subdirs)
        _aV2P_collect_buildsystem_targets("${_aV2P_subdir}" _aV2P_child)
        list(APPEND _aV2P_targets ${_aV2P_child})
    endforeach()
    list(REMOVE_DUPLICATES _aV2P_targets)
    set(${out_var} "${_aV2P_targets}" PARENT_SCOPE)
endfunction()

function(_aV2P_link_kind target out_var)
    # LINK_LIBRARIES applies to the target itself; the INTERFACE_LINK_LIBRARIES
    # of dependencies are reached transitively. Do not follow a dependency's
    # LINK_LIBRARIES, because those are private to that dependency.
    get_target_property(_aV2P_direct "${target}" LINK_LIBRARIES)
    if(NOT _aV2P_direct)
        set(_aV2P_direct "")
    endif()

    foreach(_aV2P_lib IN LISTS _aV2P_direct)
        if(_aV2P_lib MATCHES "aV2P::aV2P|^aV2P$")
            set(${out_var} "direct" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(_aV2P_queue ${_aV2P_direct})
    set(_aV2P_visited "")
    set(_aV2P_kind "none")
    while(_aV2P_queue)
        list(POP_FRONT _aV2P_queue _aV2P_lib)
        if(_aV2P_lib IN_LIST _aV2P_visited)
            continue()
        endif()
        list(APPEND _aV2P_visited "${_aV2P_lib}")
        if(_aV2P_lib MATCHES "aV2P::aV2P|^aV2P$")
            set(_aV2P_kind "transitive")
        endif()
        if(TARGET "${_aV2P_lib}")
            get_target_property(_aV2P_sub "${_aV2P_lib}" INTERFACE_LINK_LIBRARIES)
            if(_aV2P_sub)
                list(APPEND _aV2P_queue ${_aV2P_sub})
            endif()
        endif()
    endwhile()
    set(${out_var} "${_aV2P_kind}" PARENT_SCOPE)
endfunction()

function(_aV2P_finalize_target_list target_list)
    if(NOT COMMAND alpaka_get_targets)
        return()
    endif()

    set(_aV2P_unfinalized "")
    set(_aV2P_auto "")
    foreach(_aV2P_target IN LISTS target_list)
        get_target_property(_aV2P_type "${_aV2P_target}" TYPE)
        if(NOT _aV2P_type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$")
            continue()
        endif()
        get_target_property(_aV2P_sources "${_aV2P_target}" SOURCES)
        if(NOT _aV2P_sources)
            continue()
        endif()
        get_target_property(_aV2P_done "${_aV2P_target}" aV2P_FINALIZE_DONE)
        if(_aV2P_done)
            continue()
        endif()
        get_target_property(_aV2P_defs "${_aV2P_target}" COMPILE_DEFINITIONS)
        if(_aV2P_defs AND "${_aV2P_defs}" MATCHES "ALPAKA_CMAKE_TARGET_[A-Z]+_FINALIZE_CALLED")
            continue()
        endif()

        _aV2P_link_kind("${_aV2P_target}" _aV2P_kind)
        if(_aV2P_kind STREQUAL "direct")
            # A plain add_executable()/add_library() consumer that links
            # aV2P::aV2P directly must opt in with aV2P_finalize().
            list(APPEND _aV2P_unfinalized "${_aV2P_target}")
        elseif(_aV2P_kind STREQUAL "transitive")
            # The target links aV2P through an interface target (for example a
            # project helper). Finalize it automatically so that targets using a
            # helper need no extra CMake step.
            list(APPEND _aV2P_auto "${_aV2P_target}")
        endif()
    endforeach()

    foreach(_aV2P_target IN LISTS _aV2P_auto)
        aV2P_finalize("${_aV2P_target}")
    endforeach()

    if(_aV2P_unfinalized)
        list(REMOVE_DUPLICATES _aV2P_unfinalized)
        string(REPLACE ";" "\n  " _aV2P_list "${_aV2P_unfinalized}")
        message(FATAL_ERROR
            "aV2P: the following targets link aV2P/native alpaka directly but were never finalized:\n"
            "  ${_aV2P_list}\n"
            "Native alpaka 3 requires a finalize step. Call aV2P_finalize(<target>) "
            "after target_link_libraries(<target> ... aV2P::aV2P), or create the "
            "target with aV2P_add_executable()/aV2P_add_library().")
    endif()
endfunction()

function(_aV2P_finalize_discovered_targets)
    _aV2P_collect_buildsystem_targets("${CMAKE_SOURCE_DIR}" _aV2P_all_targets)
    _aV2P_finalize_target_list("${_aV2P_all_targets}")
endfunction()

function(_aV2P_process_directory_targets directory)
    get_property(_aV2P_targets DIRECTORY "${directory}" PROPERTY _aV2P_FINALIZE_TARGETS)
    if(_aV2P_targets)
        set_property(DIRECTORY "${directory}" PROPERTY _aV2P_FINALIZE_TARGETS "")
        list(REMOVE_DUPLICATES _aV2P_targets)
        foreach(_aV2P_target IN LISTS _aV2P_targets)
            if(TARGET "${_aV2P_target}")
                aV2P_finalize("${_aV2P_target}")
            endif()
        endforeach()
    endif()
    # Also auto-finalize interface-linked targets of this directory and report
    # directly linked but unfinalized ones while the directory scope is still the
    # target's own scope.
    get_property(_aV2P_here DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    _aV2P_finalize_target_list("${_aV2P_here}")
endfunction()

function(_aV2P_process_targets)
    # Per-directory deferred calls finalize targets created by the aV2P helpers.
    # The call for the top-level directory can be scheduled before the helper
    # call registers its own deferred call. Process the top-level directory queue
    # here first so a top-level target created by a source-tree consumer is not
    # reported as unfinalized.
    get_property(_aV2P_top_targets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY _aV2P_FINALIZE_TARGETS)
    if(_aV2P_top_targets)
        _aV2P_process_directory_targets("${CMAKE_SOURCE_DIR}")
    endif()
    # This final top-level sweep catches targets that were never scanned in their
    # own directory and reports directly linked but unfinalized targets.
    _aV2P_finalize_discovered_targets()
endfunction()

function(_aV2P_register_deferred_process)
    get_property(_aV2P_registered GLOBAL PROPERTY _aV2P_DEFERRED_PROCESS_REGISTERED)
    if(_aV2P_registered)
        return()
    endif()
    _aV2P_has_native_alpaka(_aV2P_native)
    if(NOT _aV2P_native)
        return()
    endif()
    set_property(GLOBAL PROPERTY _aV2P_DEFERRED_PROCESS_REGISTERED TRUE)
    # The helper-created targets are finalized in their own directory. This
    # global-only sweep runs last merely to report targets that link aV2P
    # directly without ever calling aV2P_finalize(), and to reach transitive
    # targets from directories that contain no aV2P helper call.
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL _aV2P_process_targets)
endfunction()

_aV2P_register_deferred_process()
