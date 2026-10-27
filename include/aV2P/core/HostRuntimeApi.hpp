/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! CUDA/HIP runtime access from a host-only (`AV2P_HOST_ONLY`, non-`nvcc`) translation unit.
//!
//! alpaka 3 enables its CUDA/HIP backend only when the language is detected from the compiler
//! (`__CUDACC__` / `__HIP__`), because the backend headers also contain device-side code and the
//! `<<<...>>>` launch syntax.  The CUDA/HIP *runtime* API itself is plain host C++, so a host-only
//! translation unit can name, construct and query the runtime platform, device, queue and event and
//! can allocate, copy and free device memory - it just cannot build a kernel.
//!
//! This header restores that 2.2 capability for `AV2P_HOST_ONLY`:
//!  - include the vendor runtime header and enable the alpaka 3 backend macros;
//!  - declare the device-side builtins that the (never instantiated) device code in those headers
//!    refers to, so that a host compiler can parse the headers.
//!
//! A kernel launch from a host-only translation unit stays a compile-time error: alpaka 3's launch
//! operator is guarded and reports it with a `static_assert`
//! (`alpaka/api/unifiedCudaHip/Queue.hpp`).
//!
//! Opt out with `AV2P_DISABLE_HOST_RUNTIME_API`.

#if defined(AV2P_HOST_ONLY) && !defined(__CUDACC__) && !defined(AV2P_DISABLE_HOST_RUNTIME_API)

//! `VVRRP` version encoding shared with `aV2P/core/PP.hpp` (major*10000 + minor*100 + patch).
#if !defined(AV2P_HOST_RT_VERSION)
#    define AV2P_HOST_RT_VERSION(V)                                                                                \
        ((((V) / 1000llu) % 10000llu) * 10000llu + (((V) / 10llu) % 100llu) * 100llu + ((V) % 10llu))
#endif

// ---------------------------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------------------------
#    if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !defined(AV2P_LANG_CUDA)
#        if !defined(__has_include) || __has_include(<cuda_runtime.h>)
#            include <cuda_runtime.h>
#            define AV2P_LANG_CUDA AV2P_HOST_RT_VERSION(CUDART_VERSION)
#            if !defined(ALPAKA_LANG_CUDA)
#                define ALPAKA_LANG_CUDA AV2P_HOST_RT_VERSION(CUDART_VERSION)
#            endif
#            define AV2P_HOST_CUDA_RT_API 1

// The device-side builtins below are referenced only by device function templates that a host-only
// translation unit never instantiates; declarations are enough to parse the headers.  CUDA provides
// `__host__`/`__device__`/`__global__` as empty macros for a non-nvcc compilation, so the enclosing
// function templates remain ordinary (unused) host templates.
extern "C"
{
    extern uint3 blockIdx;
    extern uint3 threadIdx;
    extern dim3 blockDim;
    extern dim3 gridDim;
    unsigned __activemask();
    void __syncthreads();
    void __syncwarp(unsigned = 0xffffffffu);
    void __threadfence();
    void __threadfence_block();
    void __threadfence_system();
    unsigned __ballot_sync(unsigned, int);
    int __any_sync(unsigned, int);
    int __all_sync(unsigned, int);
    float rsqrtf(float);
    double rsqrt(double);
    int min(int, int);
    int max(int, int);
}
template<typename T>
T __shfl_sync(unsigned, T, int, int = 32);
template<typename T>
T __shfl_down_sync(unsigned, T, unsigned, int = 32);
template<typename T>
T __shfl_up_sync(unsigned, T, unsigned, int = 32);
template<typename T>
T __shfl_xor_sync(unsigned, T, int, int = 32);
template<typename T>
int __popc(T);
template<typename T>
int __clz(T);
template<typename T>
int __ffs(T);
template<typename T, typename U>
T atomicAdd(T*, U);
template<typename T, typename U>
T atomicSub(T*, U);
template<typename T, typename U>
T atomicExch(T*, U);
template<typename T, typename U>
T atomicMin(T*, U);
template<typename T, typename U>
T atomicMax(T*, U);
template<typename T, typename U>
T atomicAnd(T*, U);
template<typename T, typename U>
T atomicOr(T*, U);
template<typename T, typename U>
T atomicXor(T*, U);
template<typename T, typename U>
T atomicInc(T*, U);
template<typename T, typename U>
T atomicDec(T*, U);
template<typename T, typename U, typename V>
T atomicCAS(T*, U, V);
#        endif
#    endif

// ---------------------------------------------------------------------------------------------
// HIP
// ---------------------------------------------------------------------------------------------
#    if defined(AV2P_ACC_GPU_HIP_ENABLED) && !defined(AV2P_LANG_HIP)
#        if !defined(__has_include) || __has_include(<hip/hip_runtime.h>)
// The runtime header requires exactly one platform macro.  ROCm is the only HIP platform aV2P is
// validated against, and no host compiler defines it, so default it here.  A CUDA-platform HIP
// build can still set `__HIP_PLATFORM_NVIDIA__` before including aV2P.
#            if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#                define __HIP_PLATFORM_AMD__
#            endif
#            include <hip/hip_runtime.h>
#            define AV2P_LANG_HIP AV2P_HOST_RT_VERSION(HIP_VERSION_MAJOR * 1000 + HIP_VERSION_MINOR)
#            if !defined(ALPAKA_LANG_HIP)
#                define ALPAKA_LANG_HIP AV2P_LANG_HIP
#            endif
#            define AV2P_HOST_HIP_RT_API 1

// The HIP runtime headers declare the grid builtins and the warp/math intrinsics only for the HIP
// compiler (`__HIP_CLANG_ONLY__`); the `hipBlockIdx_*`-style spellings are macros that never exist
// for a plain host compiler.  alpaka 3's HIP backend headers reference them in non-dependent
// contexts, so they must be declared before those headers are parsed.  As on the CUDA path the
// declarations are enough: a host-only translation unit never instantiates the device code that
// calls them, and nothing here is ever linked.
extern "C"
{
    extern unsigned hipBlockIdx_x;
    extern unsigned hipBlockIdx_y;
    extern unsigned hipBlockIdx_z;
    extern unsigned hipThreadIdx_x;
    extern unsigned hipThreadIdx_y;
    extern unsigned hipThreadIdx_z;
    extern unsigned hipBlockDim_x;
    extern unsigned hipBlockDim_y;
    extern unsigned hipBlockDim_z;
    extern unsigned hipGridDim_x;
    extern unsigned hipGridDim_y;
    extern unsigned hipGridDim_z;
    unsigned long long __ballot(int);
    unsigned __lane_id();
    int __all(int);
    int __any(int);
    void __syncthreads();
}
// Memory-scope constants used by alpaka 3's emulated HIP atomics.  The HIP headers only define
// them for the HIP compiler.
#            if !defined(__HIP_MEMORY_SCOPE_AGENT)
#                define __HIP_MEMORY_SCOPE_AGENT 1
#            endif
#            if !defined(__HIP_MEMORY_SCOPE_WORKGROUP)
#                define __HIP_MEMORY_SCOPE_WORKGROUP 0
#            endif
#            if !defined(__HIP_MEMORY_SCOPE_SYSTEM)
#                define __HIP_MEMORY_SCOPE_SYSTEM 2
#            endif
// HIP math intrinsics, mirroring the overload set that clang's HIP headers provide.  They overload
// on floating-point types, so they cannot live in the `extern "C"` block above.
float rsqrtf(float);
double rsqrt(double);
template<typename T>
T min(T, T);
template<typename T>
T max(T, T);
int min(int, int);
int max(int, int);
float min(float, float);
float max(float, float);
double min(double, double);
double max(double, double);
#        endif
#    endif

#endif
