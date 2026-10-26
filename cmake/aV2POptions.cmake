#
# Copyright 2023 Benjamin Worpitz, Erik Zenker, Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
#
# Legacy aV2P 2.2 option surface.
#
# This file is the single place that declares the renamed 2.2 cache variables.
# Every variable from the 2.2 inventory exists here with the 2.2 default, even
# when the corresponding native alpaka 3 feature is not available. The
# translation to native alpaka 3 options lives in aV2PNativeOptions.cmake and
# the machine readable table is aV2POptionMapping.csv.
#
# The file is used both while building aV2P and from the installed aV2PConfig so
# that a consumer sees the same legacy cache variables.

include_guard(GLOBAL)

include(CMakeDependentOption)

#-------------------------------------------------------------------------------
# Top-level build layout (aV2P-local, no native alpaka 3 equivalent).
option(aV2P_BUILD_EXAMPLES "Build the examples" OFF)
option(aV2P_BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(aV2P_ENABLE_WERROR "Treat all warnings as errors." OFF)
option(aV2P_INSTALL_TEST_HEADER "Install headers of the namespace aV2P::test." OFF)
cmake_dependent_option(aV2P_CHECK_HEADERS
    "Check all aV2P headers as part of the tests whether they can be compiled standalone."
    OFF BUILD_TESTING OFF)
cmake_dependent_option(aV2P_USE_INTERNAL_CATCH2
    "Use internally shipped Catch2"
    ON "BUILD_TESTING OR aV2P_BUILD_BENCHMARKS" OFF)

#-------------------------------------------------------------------------------
# Accelerator / backend selection (translated at the native alpaka 3 boundary).
option(aV2P_ACC_GPU_HIP_ENABLE "Enable the HIP back-end (all other back-ends must be disabled)" OFF)
option(aV2P_ACC_GPU_HIP_ONLY_MODE "Only back-ends using HIP can be enabled in this mode." OFF)
option(aV2P_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU back-end" OFF)
option(aV2P_ACC_GPU_CUDA_ONLY_MODE "Only back-ends using CUDA can be enabled in this mode." OFF)
option(aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU back-end" OFF)
option(aV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread back-end" OFF)
option(aV2P_ACC_CPU_B_TBB_T_SEQ_ENABLE "Enable the TBB CPU grid block back-end" OFF)
option(aV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block back-end" OFF)
option(aV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread back-end" OFF)
option(aV2P_ACC_CPU_DISABLE_ATOMIC_REF "Disable atomic_ref for CPU back-ends" OFF)
option(aV2P_ACC_SYCL_ENABLE "Enable the SYCL back-end" OFF)

#-------------------------------------------------------------------------------
# Compiler switches (same names and values as 2.2).
set(aV2P_FAST_MATH "DEFAULT" CACHE STRING "Enable fast-math")
set_property(CACHE aV2P_FAST_MATH PROPERTY STRINGS "DEFAULT;ON;OFF")
set(aV2P_FTZ "DEFAULT" CACHE STRING "Set flush to zero")
set_property(CACHE aV2P_FTZ PROPERTY STRINGS "DEFAULT;ON;OFF")
set(aV2P_RELOCATABLE_DEVICE_CODE "DEFAULT" CACHE STRING
    "Enable relocatable device code for CUDA, HIP and SYCL devices")
set_property(CACHE aV2P_RELOCATABLE_DEVICE_CODE PROPERTY STRINGS "DEFAULT;ON;OFF")
set(aV2P_CUDA_SHOW_REGISTER "DEFAULT" CACHE STRING "Show kernel registers and create device ASM")
set_property(CACHE aV2P_CUDA_SHOW_REGISTER PROPERTY STRINGS "DEFAULT;ON;OFF")
set(aV2P_CUDA_KEEP_FILES "DEFAULT" CACHE STRING
    "Keep all intermediate files that are generated during internal compilation steps")
set_property(CACHE aV2P_CUDA_KEEP_FILES PROPERTY STRINGS "DEFAULT;ON;OFF")
set(aV2P_CUDA_EXPT_EXTENDED_LAMBDA "ON" CACHE STRING
    "Enable experimental, extended host-device lambdas in CUDA with nvcc")
set_property(CACHE aV2P_CUDA_EXPT_EXTENDED_LAMBDA PROPERTY STRINGS "DEFAULT;ON;OFF")
set(aV2P_HIP_KEEP_FILES "OFF" CACHE STRING
    "Keep all intermediate files that are generated during internal compilation steps")
set_property(CACHE aV2P_HIP_KEEP_FILES PROPERTY STRINGS "DEFAULT;ON;OFF")

#-------------------------------------------------------------------------------
# Project configuration.
set(aV2P_DEBUG "0" CACHE STRING "Debug level")
set_property(CACHE aV2P_DEBUG PROPERTY STRINGS "0;1;2")
set(aV2P_CXX_STANDARD "20" CACHE STRING "C++ standard version")
set_property(CACHE aV2P_CXX_STANDARD PROPERTY STRINGS "20")
set(aV2P_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB "47" CACHE STRING
    "Kibibytes of memory to allocate for block shared memory for backends requiring static allocation")
option(aV2P_DISABLE_VENDOR_RNG
    "Disable the vendor specific random number generators (NVIDIA cuRAND, AMD rocRAND, Intel DPL)" OFF)
option(aV2P_ASSERT_ACC_ENABLE
    "Enable device side asserts. In case value is OFF device side asserts will be disabled even if NDEBUG is not defined."
    ON)
option(aV2P_CUDA_SHOW_CODELINES
    "Show kernel lines in cuda-gdb and cuda-memcheck." OFF)
set(aV2P_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with aV2P")
set_property(CACHE aV2P_USE_MDSPAN PROPERTY STRINGS "SYSTEM;FETCH;OFF")

#-------------------------------------------------------------------------------
# SYCL target selection (2.2 declared these only when SYCL was enabled; aV2P
# keeps them unconditionally so the 2.2 option surface stays complete).
option(aV2P_SYCL_ONEAPI_CPU "Enable oneAPI CPU targets for the SYCL back-end" OFF)
option(aV2P_SYCL_ONEAPI_FPGA "Enable oneAPI FPGA targets for the SYCL back-end" OFF)
option(aV2P_SYCL_ONEAPI_GPU "Enable oneAPI Intel GPU targets for the SYCL back-end" OFF)
option(aV2P_SYCL_ONEAPI_GPU_NVIDIA "Enable NVIDIA GPU targets for the SYCL back-end" OFF)
option(aV2P_SYCL_ONEAPI_GPU_AMD "Enable AMD GPU targets for the SYCL back-end" OFF)
set(aV2P_SYCL_ONEAPI_FPGA_MODE "emulation" CACHE STRING "Synthesis type for oneAPI FPGA targets")
set_property(CACHE aV2P_SYCL_ONEAPI_FPGA_MODE PROPERTY STRINGS "emulation;simulation;hardware")
set(aV2P_SYCL_ONEAPI_CPU_ISA "avx2" CACHE STRING "oneAPI CPU ISA")
set_property(CACHE aV2P_SYCL_ONEAPI_CPU_ISA PROPERTY STRINGS "sse4.2;avx;avx2;avx512")
set(aV2P_SYCL_ONEAPI_FPGA_BOARD "pac_a10" CACHE STRING "oneAPI FPGA board")
set_property(CACHE aV2P_SYCL_ONEAPI_FPGA_BOARD PROPERTY STRINGS "pac_a10;pac_s10;pac_s10_usm")
set(aV2P_SYCL_ONEAPI_FPGA_BSP "intel_a10gx_pac" CACHE STRING "oneAPI FPGA BSP")
set_property(CACHE aV2P_SYCL_ONEAPI_FPGA_BSP PROPERTY STRINGS "intel_a10gx_pac;intel_s10sx_pac")
if(NOT DEFINED aV2P_SYCL_ONEAPI_GPU_DEVICES)
    if(aV2P_SYCL_ONEAPI_GPU_NVIDIA)
        set(_aV2P_SYCL_GPU_DEVICES_DEFAULT "nvidia_gpu_sm_89")
    elseif(aV2P_SYCL_ONEAPI_GPU_AMD)
        set(_aV2P_SYCL_GPU_DEVICES_DEFAULT "amd_gpu_gfx1200")
    else()
        set(_aV2P_SYCL_GPU_DEVICES_DEFAULT "intel_gpu_pvc")
    endif()
    set(aV2P_SYCL_ONEAPI_GPU_DEVICES "${_aV2P_SYCL_GPU_DEVICES_DEFAULT}" CACHE STRING
        "oneAPI GPU devices")
    unset(_aV2P_SYCL_GPU_DEVICES_DEFAULT)
endif()
