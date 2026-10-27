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
// HIP (the HIP headers are host-compilable by design and already provide the builtins)
// ---------------------------------------------------------------------------------------------
#    if defined(AV2P_ACC_GPU_HIP_ENABLED) && !defined(AV2P_LANG_HIP)
#        if !defined(__has_include) || __has_include(<hip/hip_runtime.h>)
#            include <hip/hip_runtime.h>
#            define AV2P_LANG_HIP AV2P_HOST_RT_VERSION(HIP_VERSION_MAJOR * 1000 + HIP_VERSION_MINOR)
#            if !defined(ALPAKA_LANG_HIP)
#                define ALPAKA_LANG_HIP AV2P_LANG_HIP
#            endif
#            define AV2P_HOST_HIP_RT_API 1
#        endif
#    endif

#endif
