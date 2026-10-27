/* Copyright 2023 Benjamin Worpitz, Matthias Werner, René Widera, Sergei Bastrakov, Jeffrey Kelling,
 *                Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/PP.hpp"
// Enables the CUDA/HIP runtime backends for host-only (`AV2P_HOST_ONLY`) translation units.
#include "aV2P/core/HostRuntimeApi.hpp"

#ifdef __INTEL_COMPILER
#    warning                                                                                                          \
        "The Intel Classic compiler (icpc) is no longer supported. Please upgrade to the Intel LLVM compiler (ipcx)."
#endif

// ######## detect operating systems ########

// WINDOWS
#if !defined(AV2P_OS_WINDOWS)
#    if defined(_WIN64) || defined(__MINGW64__)
#        define AV2P_OS_WINDOWS 1
#    else
#        define AV2P_OS_WINDOWS 0
#    endif
#endif


// Linux
#if !defined(AV2P_OS_LINUX)
#    if defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
#        define AV2P_OS_LINUX 1
#    else
#        define AV2P_OS_LINUX 0
#    endif
#endif

// Apple
#if !defined(AV2P_OS_IOS)
#    if defined(__APPLE__)
#        define AV2P_OS_IOS 1
#    else
#        define AV2P_OS_IOS 0
#    endif
#endif

// Cygwin
#if !defined(AV2P_OS_CYGWIN)
#    if defined(__CYGWIN__)
#        define AV2P_OS_CYGWIN 1
#    else
#        define AV2P_OS_CYGWIN 0
#    endif
#endif

// ### architectures

// X86
#if !defined(AV2P_ARCH_X86)
#    if defined(__x86_64__) || defined(_M_X64)
#        define AV2P_ARCH_X86 1
#    else
#        define AV2P_ARCH_X86 0
#    endif
#endif

// RISCV
#if !defined(AV2P_ARCH_RISCV)
#    if defined(__riscv)
#        define AV2P_ARCH_RISCV 1
#    else
#        define AV2P_ARCH_RISCV 0
#    endif
#endif

// ARM
#if !defined(AV2P_ARCH_ARM)
#    if defined(__ARM_ARCH) || defined(__arm__) || defined(__arm64)
#        define AV2P_ARCH_ARM 1
#    else
#        define AV2P_ARCH_ARM 0
#    endif
#endif

// NVIDIA device compile
// AV2P_ARCH_PTX is a misnomer as it holds the compute capability version (the virtual GPU architecture) and not the
// PTX ISA version. This misnomer exists in CUDA as well. https://github.com/NVIDIA/cccl/issues/4081
#if !defined(AV2P_ARCH_PTX)
#    if defined(__CUDA_ARCH__)
#        define AV2P_ARCH_PTX AV2P_VRP_TO_VERSION(__CUDA_ARCH__)
#    else
#        define AV2P_ARCH_PTX AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

/** HIP device compile
 *
 * The version on the host side will always be AV2P_VERSION_NUMBER_NOT_AVAILABLE.
 * On the device side unknown version will be set to AV2P_VERSION_NUMBER_NOT_AVAILABLE.
 *
 *  Rules:
 *   - the last two digits will be handled as HEX values and support 0-9 and a-f
 *   - gfx9xy (numeric): 9xy -> AV2P_VERSION_NUMBER(9,x,y)
 *   - gfx10xy / gfx11xy: stxy -> AV2P_VERSION_NUMBER(st,x,y)
 *   - Suffix: a == 10, b == 11, c == 12
 *      - gfx90a -> AV2P_VERSION_NUMBER(9,0,10)
 *      - gfx90c -> AV2P_VERSION_NUMBER(9,0,12)
 */
#if !defined(AV2P_ARCH_AMD)
#    if defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
#        include "aV2P/core/HipConfig.hpp"
#        define AV2P_ARCH_AMD AV2P_AMDGPU_ARCH
#    else
#        define AV2P_ARCH_AMD AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

/** HSA device compile
 *
 * This version is equal to AV2P_ARCH_AMD and is only carried for backwards compatibility to older ALPAKA versions.
 */
#define AV2P_ARCH_HSA AV2P_ARCH_AMD

// ######## compiler ########

// HIP compiler detection
#if !defined(AV2P_COMP_HIP)
#    if defined(__HIP__) // Defined by hip-clang and vanilla clang in HIP mode.
#        include <hip/hip_version.h>
// HIP doesn't give us a patch level for the last entry, just a gitdate
#        define AV2P_COMP_HIP AV2P_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#    else
#        define AV2P_COMP_HIP AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// nvcc compiler
#if defined(__NVCC__)
#    define AV2P_COMP_NVCC AV2P_VERSION_NUMBER(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__)
#else
#    define AV2P_COMP_NVCC AV2P_VERSION_NUMBER_NOT_AVAILABLE
#endif

// clang compiler
#if defined(__clang__)
#    define AV2P_COMP_CLANG AV2P_VERSION_NUMBER(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#    define AV2P_COMP_CLANG AV2P_VERSION_NUMBER_NOT_AVAILABLE
#endif

// MSVC compiler
#if defined(_MSC_VER)
#    define AV2P_COMP_MSVC                                                                                            \
        AV2P_VERSION_NUMBER((_MSC_FULL_VER) % 10'000'000, ((_MSC_FULL_VER) / 100000) % 100, (_MSC_FULL_VER) % 100000)
#else
#    define AV2P_COMP_MSVC AV2P_VERSION_NUMBER_NOT_AVAILABLE
#endif

// gnu compiler (excluding compilers which emulates gnu compiler like clang)
#if defined(__GNUC__) && !defined(__clang__)
#    if defined(__GNUC_PATCHLEVEL__)
#        define AV2P_COMP_GNUC AV2P_VERSION_NUMBER(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#    else
#        define AV2P_COMP_GNUC AV2P_VERSION_NUMBER(__GNUC__, __GNUC_MINOR__, 0)
#    endif
#else
#    define AV2P_COMP_GNUC AV2P_VERSION_NUMBER_NOT_AVAILABLE
#endif

// IBM compiler
// only clang based is supported
#if defined(__ibmxl__)
#    define AV2P_COMP_IBM AV2P_VERSION_NUMBER(__ibmxl_version__, __ibmxl_release__, __ibmxl_modification__)
#else
#    define AV2P_COMP_IBM AV2P_VERSION_NUMBER_NOT_AVAILABLE
#endif

// clang CUDA compiler detection
// Currently __CUDA__ is only defined by clang when compiling CUDA code.
#if defined(__clang__) && defined(__CUDA__)
#    define AV2P_COMP_CLANG_CUDA AV2P_VERSION_NUMBER(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#    define AV2P_COMP_CLANG_CUDA AV2P_VERSION_NUMBER_NOT_AVAILABLE
#endif

// PGI and NV HPC SDK compiler detection
#if defined(__PGI)
#    define AV2P_COMP_PGI AV2P_VERSION_NUMBER(__PGIC__, __PGIC_MINOR__, __PGIC_PATCHLEVEL__)
#else
#    define AV2P_COMP_PGI AV2P_VERSION_NUMBER_NOT_AVAILABLE
#endif

// Intel LLVM compiler detection
#if !defined(AV2P_COMP_ICPX)
#    if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
// The version string for icpx 2023.1.0 is 20230100. In Boost.Predef this becomes (53,1,0).
#        define AV2P_COMP_ICPX AV2P_YYYYMMDD_TO_VERSION(__INTEL_LLVM_COMPILER)
#    else
#        define AV2P_COMP_ICPX AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// ######## C++ language ########

//---------------------------------------HIP-----------------------------------
// __HIP__ is defined by both hip-clang and vanilla clang in HIP mode.
// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md#compiler-defines-summary
#if !defined(AV2P_LANG_HIP)
#    if defined(__HIP__)
#        include <hip/hip_version.h>
// HIP doesn't give us a patch level for the last entry, just a gitdate
#        define AV2P_LANG_HIP AV2P_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#    else
#        define AV2P_LANG_HIP AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// CUDA
#if !defined(AV2P_LANG_CUDA)
#    if defined(__CUDACC__) || defined(__CUDA__)
#        include <cuda.h>
// CUDA doesn't give us a patch level for the last entry, just zero.
#        define AV2P_LANG_CUDA AV2P_VVRRP_TO_VERSION(CUDART_VERSION)
#    else
#        define AV2P_LANG_CUDA AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// Intel OneAPI Sycl GPU
#if !defined(AV2P_LANG_SYCL)
#    if defined(SYCL_LANGUAGE_VERSION)
#        define AV2P_LANG_SYCL AV2P_YYYYMMDD_TO_VERSION(SYCL_LANGUAGE_VERSION)
#    else
#        define AV2P_LANG_SYCL AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#    if(AV2P_COMP_ICPX)
// ONE API must be detected via the ICPX compiler see
// https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/use-predefined-macros-to-specify-intel-compilers.html
#        define AV2P_LANG_ONEAPI AV2P_COMP_ICPX
#    endif
#endif

// OpenMP
#if !defined(AV2P_OMP)
#    if defined(_OPENMP)
#        include <omp.h>
#    endif
#    if defined(_OPENMP)
#        define AV2P_OMP AV2P_YYYYMM_TO_VERSION(_OPENMP)
#    else
#        define AV2P_OMP AV2P_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif
