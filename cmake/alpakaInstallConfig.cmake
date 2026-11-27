#
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#

# Generate the installed configuration for a pinned alpaka package.
#
# This file is only used while configuring the installation. It reads the effective
# alpaka_DEP_* and CMAKE_* values and fills cmake/alpakaConfigPinned.cmake.in.
# The generated file is installed next to alpakaConfig.cmake.

function(alpaka_generate_install_config output)
    set(_alpaka_accelerator_pinned OFF)
    if(alpaka_DEP_CUDA OR alpaka_DEP_HIP OR alpaka_DEP_ONEAPI)
        set(_alpaka_accelerator_pinned ON)
    endif()

    set(ALPAKA_PIN_ACCELERATOR ${_alpaka_accelerator_pinned})
    set(ALPAKA_PIN_OMP ${alpaka_DEP_OMP})
    set(ALPAKA_PIN_HWLOC ${alpaka_DEP_HWLOC})
    set(ALPAKA_PIN_CUDA ${alpaka_DEP_CUDA})
    set(ALPAKA_PIN_HIP ${alpaka_DEP_HIP})
    set(ALPAKA_PIN_ONEAPI ${alpaka_DEP_ONEAPI})

    set(ALPAKA_PIN_HWLOC_PREFIX "")
    set(ALPAKA_PIN_HWLOC_PKGCONFIG "")
    if(alpaka_DEP_HWLOC STREQUAL "ON")
        set(_alpaka_hwloc_prefix "")
        if(DEFINED HWLOC_PREFIX AND HWLOC_PREFIX)
            set(_alpaka_hwloc_prefix "${HWLOC_PREFIX}")
        elseif(DEFINED HWLOC_INCLUDE_DIRS AND HWLOC_INCLUDE_DIRS)
            list(GET HWLOC_INCLUDE_DIRS 0 _alpaka_hwloc_include_dir)
            get_filename_component(_alpaka_hwloc_prefix "${_alpaka_hwloc_include_dir}" DIRECTORY)
        elseif(DEFINED HWLOC_LIBRARY_DIRS AND HWLOC_LIBRARY_DIRS)
            list(GET HWLOC_LIBRARY_DIRS 0 _alpaka_hwloc_library_dir)
            get_filename_component(_alpaka_hwloc_prefix "${_alpaka_hwloc_library_dir}" DIRECTORY)
        endif()

        if(_alpaka_hwloc_prefix)
            file(REAL_PATH "${_alpaka_hwloc_prefix}" _alpaka_hwloc_prefix)
            set(ALPAKA_PIN_HWLOC_PREFIX "${_alpaka_hwloc_prefix}")

            foreach(_alpaka_pkgconfig_suffix lib/pkgconfig lib64/pkgconfig)
                set(_alpaka_hwloc_pkgconfig "${_alpaka_hwloc_prefix}/${_alpaka_pkgconfig_suffix}")
                if(EXISTS "${_alpaka_hwloc_pkgconfig}")
                    set(ALPAKA_PIN_HWLOC_PKGCONFIG "${_alpaka_hwloc_pkgconfig}")
                    break()
                endif()
            endforeach()
        endif()
    endif()

    set(ALPAKA_PIN_CUDA_COMPILER "")
    set(ALPAKA_PIN_CUDA_COMPILER_ID "")
    set(ALPAKA_PIN_CUDA_TOOLKIT_ROOT "")
    set(ALPAKA_PIN_CUDA_ARCHITECTURES "")
    if(alpaka_DEP_CUDA)
        if(NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR "CUDA is pinned but CMake did not enable the CUDA language.")
        endif()

        set(ALPAKA_PIN_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
        set(ALPAKA_PIN_CUDA_COMPILER_ID "${CMAKE_CUDA_COMPILER_ID}")

        if(DEFINED CUDAToolkit_ROOT AND CUDAToolkit_ROOT)
            set(_alpaka_cuda_toolkit_root "${CUDAToolkit_ROOT}")
        elseif(DEFINED CMAKE_CUDA_COMPILER_TOOLKIT_ROOT AND CMAKE_CUDA_COMPILER_TOOLKIT_ROOT)
            set(_alpaka_cuda_toolkit_root "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
        endif()
        if(_alpaka_cuda_toolkit_root)
            file(REAL_PATH "${_alpaka_cuda_toolkit_root}" ALPAKA_PIN_CUDA_TOOLKIT_ROOT)
        elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "A Clang CUDA installation requires a CUDA toolkit root. Set CUDAToolkit_ROOT.")
        endif()

        if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
            set(ALPAKA_PIN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()

    set(ALPAKA_PIN_HIP_COMPILER "")
    set(ALPAKA_PIN_HIP_DIR "")
    set(ALPAKA_PIN_HIP_ROOT "")
    set(ALPAKA_PIN_HIP_ARCHITECTURES "")
    if(alpaka_DEP_HIP)
        if(NOT CMAKE_HIP_COMPILER)
            message(FATAL_ERROR "HIP is pinned but CMake did not enable the HIP language.")
        endif()

        set(ALPAKA_PIN_HIP_COMPILER "${CMAKE_HIP_COMPILER}")

        if(DEFINED hip_DIR AND hip_DIR)
            file(REAL_PATH "${hip_DIR}" ALPAKA_PIN_HIP_DIR)
            get_filename_component(ALPAKA_PIN_HIP_ROOT "${ALPAKA_PIN_HIP_DIR}/../../.." ABSOLUTE)
        endif()

        if(DEFINED CMAKE_HIP_ARCHITECTURES AND CMAKE_HIP_ARCHITECTURES)
            set(ALPAKA_PIN_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
        endif()
    endif()

    set(ALPAKA_PIN_ONEAPI_COMPILER "")
    set(ALPAKA_PIN_ONEAPI_INTELSYCL_DIR "")
    set(ALPAKA_PIN_ONEAPI_PREFIX "")
    if(alpaka_DEP_ONEAPI)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            message(
                FATAL_ERROR
                "oneAPI is pinned but the CXX compiler is '${CMAKE_CXX_COMPILER_ID}'. "
                "Configure the installation with the IntelLLVM compiler (icpx)."
            )
        endif()

        set(ALPAKA_PIN_ONEAPI_COMPILER "${CMAKE_CXX_COMPILER}")
        if(NOT DEFINED IntelSYCL_DIR OR NOT IntelSYCL_DIR)
            message(
                FATAL_ERROR
                "oneAPI is pinned but IntelSYCL_DIR is not set. The IntelSYCL CMake package is required."
            )
        endif()
        set(ALPAKA_PIN_ONEAPI_INTELSYCL_DIR "${IntelSYCL_DIR}")

        get_filename_component(_alpaka_oneapi_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ALPAKA_PIN_ONEAPI_PREFIX "${_alpaka_oneapi_bin}" DIRECTORY)
    endif()

    configure_file("${_alpaka_CMAKE_DIR}/alpakaConfigPinned.cmake.in" "${output}" @ONLY)
endfunction()
