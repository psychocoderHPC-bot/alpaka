/* Copyright 2022 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/atomic/Op.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/core/Utility.hpp"

#include <type_traits>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP accelerator atomic ops.
    //
    //  Atomics can be used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be safe between devices.
    class AtomicUniformCudaHipBuiltIn
    {
    };
} // namespace aV2P

#    if !defined(AV2P_HOST_ONLY)

#        if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA
#            error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP
#            error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

//! clang is providing a builtin for different atomic functions even if these is not supported for architectures < 6.0
#        define CLANG_CUDA_PTX_WORKAROUND                                                                             \
            (AV2P_COMP_CLANG && AV2P_LANG_CUDA && AV2P_ARCH_PTX < AV2P_VERSION_NUMBER(6, 0, 0))

//! These types must be in the global namespace for checking existence of respective functions in global namespace via
//! SFINAE, so we use inline namespace.
inline namespace aV2PGlobal
{
    //! Provide an interface to builtin atomic functions.
    //
    // To check for the existence of builtin functions located in the global namespace :: directly.
    // This would not be possible without having these types in global namespace.
    // If the functor is inheriting from std::false_type an signature is explicitly not available. This can be used to
    // explicitly disable builtin function in case the builtin is broken.
    // If the functor is inheriting from std::true_type a specialization must implement one of the following
    // interfaces.
    // \code{.cpp}
    //    // interface for all atomics except atomicCas
    //    __device__ static T atomic( T* add, T value);
    //    // interface for atomicCas only
    //    __device__ static T atomic( T* add, T compare, T value);
    // \endcode
    template<typename TOp, typename T, typename THierarchy, typename TSfinae = void>
    struct AlpakaBuiltInAtomic : std::false_type
    {
    };

    // Cas.
    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicCas,
        T,
        THierarchy,
        typename std::void_t<
            decltype(atomicCAS(aV2P::core::declval<T*>(), aV2P::core::declval<T>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T compare, T value)
        {
            return atomicCAS(add, compare, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicCas,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicCAS_block(
            aV2P::core::declval<T*>(),
            aV2P::core::declval<T>(),
            aV2P::core::declval<T>()))>> : std::true_type
    {
        static __device__ T atomic(T* add, T compare, T value)
        {
            return atomicCAS_block(add, compare, value);
        }
    };
#        endif


    // Add.
    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicAdd,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicAdd(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAdd(add, value);
        }
    };


#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicAdd,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicAdd_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAdd_block(add, value);
        }
    };
#        endif

#        if CLANG_CUDA_PTX_WORKAROUND
    // clang is providing a builtin for atomicAdd even if these is not supported by the current architecture
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<aV2P::AtomicAdd, double, THierarchy> : std::false_type
    {
    };
#        endif

#        if(AV2P_LANG_HIP)
    // HIP shows bad performance with builtin atomicAdd(float*,float) for the hierarchy threads therefore we do not
    // call the buildin method and instead use the atomicCAS emulation. For details see:
    // https://github.com/alpaka-group/alpaka/issues/1657
    template<>
    struct AlpakaBuiltInAtomic<aV2P::AtomicAdd, float, aV2P::hierarchy::Threads> : std::false_type
    {
    };
#        endif

    // Sub.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicSub,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicSub(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicSub(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicSub,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicSub_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicSub_block(add, value);
        }
    };
#        endif

    // Min.
    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicMin,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicMin(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMin(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicMin,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicMin_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMin_block(add, value);
        }
    };
#        endif

// disable HIP atomicMin: see https://github.com/ROCm-Developer-Tools/hipamd/pull/40
#        if(AV2P_LANG_HIP)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMin, float, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMin, float, aV2P::hierarchy::Threads> : std::false_type
    {
    };

    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMin, double, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMin, double, aV2P::hierarchy::Threads> : std::false_type
    {
    };

#            if !__has_builtin(__hip_atomic_compare_exchange_strong)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMin, unsigned long long, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMin, unsigned long long, aV2P::hierarchy::Threads> : std::false_type
    {
    };
#            endif
#        endif

    // Max.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicMax,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicMax(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMax(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicMax,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicMax_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicMax_block(add, value);
        }
    };
#        endif

    // disable HIP atomicMax: see https://github.com/ROCm-Developer-Tools/hipamd/pull/40
#        if(AV2P_LANG_HIP)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMax, float, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMax, float, aV2P::hierarchy::Threads> : std::false_type
    {
    };

    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMax, double, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMax, double, aV2P::hierarchy::Threads> : std::false_type
    {
    };

#            if !__has_builtin(__hip_atomic_compare_exchange_strong)
    template<typename THierarchy>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMax, unsigned long long, THierarchy> : std::false_type
    {
    };

    template<>
    struct AlpakaBuiltInAtomic<aV2P::AtomicMax, unsigned long long, aV2P::hierarchy::Threads> : std::false_type
    {
    };
#            endif
#        endif


    // Exch.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicExch,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicExch(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicExch(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicExch,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicExch_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicExch_block(add, value);
        }
    };
#        endif

    // Inc.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicInc,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicInc(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicInc(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicInc,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicInc_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicInc_block(add, value);
        }
    };
#        endif

    // Dec.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicDec,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicDec(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicDec(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicDec,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicDec_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicDec_block(add, value);
        }
    };
#        endif

    // And.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicAnd,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicAnd(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAnd(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicAnd,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicAnd_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicAnd_block(add, value);
        }
    };
#        endif

    // Or.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicOr,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicOr(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicOr(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicOr,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicOr_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicOr_block(add, value);
        }
    };
#        endif

    // Xor.

    template<typename T, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicXor,
        T,
        THierarchy,
        typename std::void_t<decltype(atomicXor(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicXor(add, value);
        }
    };

#        if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T>
    struct AlpakaBuiltInAtomic<
        aV2P::AtomicXor,
        T,
        aV2P::hierarchy::Threads,
        typename std::void_t<decltype(atomicXor_block(aV2P::core::declval<T*>(), aV2P::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic(T* add, T value)
        {
            return atomicXor_block(add, value);
        }
    };
#        endif

} // namespace aV2PGlobal

#        undef CLANG_CUDA_PTX_WORKAROUND
#    endif

#endif
