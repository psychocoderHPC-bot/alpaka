/* Copyright 2022 Benjamin Worpitz, René Widera, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/atomic/AtomicUniformCudaHip.hpp"
#include "aV2P/atomic/Op.hpp"
#include "aV2P/atomic/Traits.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Decay.hpp"
#include "aV2P/core/Unreachable.hpp"

#include <bit>
#include <limits>
#include <type_traits>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

#    if !defined(AV2P_HOST_ONLY)

#        if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA
#            error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP
#            error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

namespace aV2P::trait
{
    namespace detail
    {
        struct EmulationBase
        {
            template<typename T>
            using AtomicCasType = std::conditional_t<
                sizeof(T) == 4u,
                unsigned int,
                std::conditional_t<sizeof(T) == 8u, unsigned long long int, void>>;

            template<typename T>
            static __device__ auto reinterpretAddress(T* address)
                -> AtomicCasType<T>* requires(sizeof(T) == 4u || sizeof(T) == 8u) {
                    return reinterpret_cast<AtomicCasType<T>*>(address);
                }

            template<typename T>
            static __device__ auto reinterpretValue(T value)
                -> AtomicCasType<T> requires(sizeof(T) == 4u || sizeof(T) == 8u)
            {
                return std::bit_cast<AtomicCasType<T>>(value);
            }
        };

        //! Emulate atomic
        //
        // The default implementation will emulate all atomic functions with atomicCAS.
        template<
            typename TOp,
            typename TAtomic,
            typename T,
            typename THierarchy,
            typename TSfinae = void,
            typename TDefer = void>
        struct EmulateAtomic : private EmulationBase
        {
        public:
            static __device__ auto atomic(
                aV2P::AtomicUniformCudaHipBuiltIn const& ctx,
                T* const addr,
                T const& value) -> T
            {
                auto* const addressAsIntegralType = reinterpretAddress(addr);
                using EmulatedType = std::decay_t<decltype(*addressAsIntegralType)>;

// Emulating atomics with atomicCAS is mentioned in the programming guide too.
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
#        if AV2P_LANG_HIP
#            if __has_builtin(__scoped_atomic_load_n)
                EmulatedType old{
                    __scoped_atomic_load_n(addressAsIntegralType, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)};
#            elif __has_builtin(__hip_atomic_load)
                EmulatedType old{__hip_atomic_load(addressAsIntegralType, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)};
#            else
                EmulatedType old{__atomic_load_n(addressAsIntegralType, __ATOMIC_RELAXED)};
#            endif
#        else
                EmulatedType old{*addressAsIntegralType};
#        endif
                EmulatedType assumed;
                do
                {
                    assumed = old;
                    T v = std::bit_cast<T>(assumed);
                    TOp{}(&v, value);
                    using Cas = aV2P::trait::
                        AtomicOp<aV2P::AtomicCas, aV2P::AtomicUniformCudaHipBuiltIn, EmulatedType, THierarchy>;
                    old = Cas::atomicOp(ctx, addressAsIntegralType, assumed, reinterpretValue(v));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                } while(assumed != old);
                return std::bit_cast<T>(old);
            }
        };

        //! Emulate AtomicCas with equivalent unisigned integral type
        template<typename T, typename THierarchy>
        struct EmulateAtomic<aV2P::AtomicCas, aV2P::AtomicUniformCudaHipBuiltIn, T, THierarchy>
            : private EmulationBase
        {
            static __device__ auto atomic(
                aV2P::AtomicUniformCudaHipBuiltIn const& ctx,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                auto* const addressAsIntegralType = reinterpretAddress(addr);
                using EmulatedType = std::decay_t<decltype(*addressAsIntegralType)>;
                EmulatedType reinterpretedCompare = reinterpretValue(compare);
                EmulatedType reinterpretedValue = reinterpretValue(value);

                auto old = aV2P::trait::
                    AtomicOp<aV2P::AtomicCas, aV2P::AtomicUniformCudaHipBuiltIn, EmulatedType, THierarchy>::
                        atomicOp(ctx, addressAsIntegralType, reinterpretedCompare, reinterpretedValue);

                return std::bit_cast<T>(old);
            }
        };

        //! Emulate AtomicSub with atomicAdd
        template<typename T, typename THierarchy>
        struct EmulateAtomic<aV2P::AtomicSub, aV2P::AtomicUniformCudaHipBuiltIn, T, THierarchy>
        {
            static __device__ auto atomic(
                aV2P::AtomicUniformCudaHipBuiltIn const& ctx,
                T* const addr,
                T const& value) -> T
            {
                return aV2P::trait::AtomicOp<aV2P::AtomicAdd, aV2P::AtomicUniformCudaHipBuiltIn, T, THierarchy>::
                    atomicOp(ctx, addr, -value);
            }
        };

        //! AtomicDec can not be implemented for floating point types!
        template<typename T, typename THierarchy>
        struct EmulateAtomic<
            aV2P::AtomicDec,
            aV2P::AtomicUniformCudaHipBuiltIn,
            T,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(aV2P::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<aV2P::AtomicDec> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicInc can not be implemented for floating point types!
        template<typename T, typename THierarchy>
        struct EmulateAtomic<
            aV2P::AtomicInc,
            aV2P::AtomicUniformCudaHipBuiltIn,
            T,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(aV2P::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<aV2P::AtomicInc> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicAnd can not be implemented for floating point types!
        template<typename T, typename THierarchy>
        struct EmulateAtomic<
            aV2P::AtomicAnd,
            aV2P::AtomicUniformCudaHipBuiltIn,
            T,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(aV2P::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<aV2P::AtomicAnd> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicOr can not be implemented for floating point types!
        template<typename T, typename THierarchy>
        struct EmulateAtomic<
            aV2P::AtomicOr,
            aV2P::AtomicUniformCudaHipBuiltIn,
            T,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(aV2P::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<aV2P::AtomicOr> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicXor can not be implemented for floating point types!
        template<typename T, typename THierarchy>
        struct EmulateAtomic<
            aV2P::AtomicXor,
            aV2P::AtomicUniformCudaHipBuiltIn,
            T,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(aV2P::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<aV2P::AtomicXor> is not supported for floating point data types!");
                return T{};
            }
        };

    } // namespace detail

    //! Generic atomic implementation
    //
    // - unsigned long int will be redirected to unsigned long long int or unsigned int implementation depending if
    //   unsigned long int is a 64 or 32bit data type.
    // - Atomics which are not available as builtin atomic will be emulated.
    template<typename TOp, typename T, typename THierarchy>
    struct AtomicOp<TOp, AtomicUniformCudaHipBuiltIn, T, THierarchy>
    {
        static __device__ auto atomicOp(
            AtomicUniformCudaHipBuiltIn const& ctx,
            [[maybe_unused]] T* const addr,
            [[maybe_unused]] T const& value) -> T
        {
            static_assert(
                sizeof(T) == 4u || sizeof(T) == 8u,
                "atomicOp<TOp, AtomicUniformCudaHipBuiltIn, T>(atomic, addr, value) is not supported! Only 64 and "
                "32bit atomics are supported.");

            if constexpr(::AlpakaBuiltInAtomic<TOp, T, THierarchy>::value)
                return ::AlpakaBuiltInAtomic<TOp, T, THierarchy>::atomic(addr, value);

            else if constexpr(std::is_same_v<unsigned long int, T>)
            {
                if constexpr(sizeof(T) == 4u && ::AlpakaBuiltInAtomic<TOp, unsigned int, THierarchy>::value)
                    return ::AlpakaBuiltInAtomic<TOp, unsigned int, THierarchy>::atomic(
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(value));
                else if constexpr(
                    sizeof(T) == 8u && ::AlpakaBuiltInAtomic<TOp, unsigned long long int, THierarchy>::value) // LP64
                {
                    return ::AlpakaBuiltInAtomic<TOp, unsigned long long int, THierarchy>::atomic(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));
                }
            }

            return detail::EmulateAtomic<TOp, AtomicUniformCudaHipBuiltIn, T, THierarchy>::atomic(ctx, addr, value);
        }
    };

    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, T, THierarchy>
    {
        static __device__ auto atomicOp(
            [[maybe_unused]] AtomicUniformCudaHipBuiltIn const& ctx,
            [[maybe_unused]] T* const addr,
            [[maybe_unused]] T const& compare,
            [[maybe_unused]] T const& value) -> T
        {
            static_assert(
                sizeof(T) == 4u || sizeof(T) == 8u,
                "atomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, T>(atomic, addr, compare, value) is not "
                "supported! Only 64 and "
                "32bit atomics are supported.");

            if constexpr(::AlpakaBuiltInAtomic<AtomicCas, T, THierarchy>::value)
                return ::AlpakaBuiltInAtomic<AtomicCas, T, THierarchy>::atomic(addr, compare, value);

            else if constexpr(std::is_same_v<unsigned long int, T>)
            {
                if constexpr(sizeof(T) == 4u && ::AlpakaBuiltInAtomic<AtomicCas, unsigned int, THierarchy>::value)
                    return ::AlpakaBuiltInAtomic<AtomicCas, unsigned int, THierarchy>::atomic(
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(compare),
                        static_cast<unsigned int>(value));
                else if constexpr(
                    sizeof(T) == 8u
                    && ::AlpakaBuiltInAtomic<AtomicCas, unsigned long long int, THierarchy>::value) // LP64
                {
                    return ::AlpakaBuiltInAtomic<AtomicCas, unsigned long long int, THierarchy>::atomic(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(compare),
                        static_cast<unsigned long long int>(value));
                }
            }

            return detail::EmulateAtomic<AtomicCas, AtomicUniformCudaHipBuiltIn, T, THierarchy>::atomic(
                ctx,
                addr,
                compare,
                value);
        }
    };
} // namespace aV2P::trait
#    endif
#endif
