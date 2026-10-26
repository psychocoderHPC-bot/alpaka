/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
                  Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/elem/Traits.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/meta/Concatenate.hpp"
#include "aV2P/meta/TypeListOps.hpp"
#include "aV2P/offset/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

#include <tuple>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

#    ifdef AV2P_ACC_GPU_CUDA_ENABLED
#        include <cuda.h>
#        include <cuda_runtime.h>
#        if __has_include(<cuda/atomic>)
#            define AV2P_CUDA_ATOMIC
#            include <cuda/atomic>
#            if AV2P_COMP_CLANG_CUDA && defined(_Float16)
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wreserved-identifier"
// We see errors when using clang as the CUDA compiler if TBB is also enabled
// Errors occour inside TBB because the _Float16 macro is redefined and pulled in from <cuda/atomic>
#                undef _Float16
#                pragma clang diagnostic pop
#            endif
#        endif
#        ifdef AV2P_USE_MDSPAN
// needs to be included that cuda::std::numeric_limits is defined and in the correct namespace
#            include "aV2P/core/mdspan.hpp"
#        endif
#    endif

#    ifdef AV2P_ACC_GPU_HIP_ENABLED
#        if AV2P_COMP_HIP >= AV2P_VERSION_NUMBER(6, 2, 0) && AV2P_COMP_HIP < AV2P_VERSION_NUMBER(7, 0, 0)
#            define HIP_ENABLE_WARP_SYNC_BUILTINS
#        endif
#        include <hip/hip_runtime.h>
#    endif

namespace aV2P
{
    namespace detail
    {
        using CudaHipBuiltinTypes1 = std::
            tuple<char1, double1, float1, int1, long1, longlong1, short1, uchar1, uint1, ulong1, ulonglong1, ushort1>;
        using CudaHipBuiltinTypes2 = std::
            tuple<char2, double2, float2, int2, long2, longlong2, short2, uchar2, uint2, ulong2, ulonglong2, ushort2>;
        using CudaHipBuiltinTypes3 = std::tuple<
            char3,
            dim3,
            double3,
            float3,
            int3,
            long3,
            longlong3,
            short3,
            uchar3,
            uint3,
            ulong3,
            ulonglong3,
            ushort3
// CUDA built-in variables have special types in clang native CUDA compilation
// defined in cuda_builtin_vars.h
#    if AV2P_COMP_CLANG_CUDA
            ,
            __cuda_builtin_threadIdx_t,
            __cuda_builtin_blockIdx_t,
            __cuda_builtin_blockDim_t,
            __cuda_builtin_gridDim_t
#    endif
            >;
        using CudaHipBuiltinTypes4 = std::tuple<
            char4,
            float4,
            int4,
            short4,
            uchar4,
            uint4,
            ushort4,
        // double4, long4, longlong4, ulong4, ulonglong4 is deprecated in
        // CUDA 13.0 and will be removed in CUDA 14.0
#    if defined(CUDART_VERSION) && (AV2P_VVRRP_TO_VERSION(CUDART_VERSION) >= AV2P_VERSION_NUMBER(13, 0, 0))
            double4_16a,
            double4_32a,
            long4_16a,
            long4_32a,
            longlong4_16a,
            longlong4_32a,
            ulong4_16a,
            ulong4_32a,
            ulonglong4_16a,
            ulonglong4_32a
#    else
            double4,
            long4,
            longlong4,
            ulong4,
            ulonglong4
#    endif
            >;

        using CudaHipBuiltinTypes = meta::
            Concatenate<CudaHipBuiltinTypes1, CudaHipBuiltinTypes2, CudaHipBuiltinTypes3, CudaHipBuiltinTypes4>;

        template<typename T>
        inline constexpr auto isCudaHipBuiltInType = meta::Contains<CudaHipBuiltinTypes, T>::value;
    } // namespace detail

#    ifdef AV2P_ACC_GPU_CUDA_ENABLED
    namespace cuda::trait
    {
        template<typename T>
        inline constexpr auto isCudaBuiltInType = aV2P::detail::isCudaHipBuiltInType<T>;
    } // namespace cuda::trait
#    endif

#    ifdef AV2P_ACC_GPU_HIP_ENABLED
    namespace hip::trait
    {
        template<typename T>
        inline constexpr auto isHipBuiltInType = aV2P::detail::isCudaHipBuiltInType<T>;
    } // namespace hip::trait
#    endif

    namespace trait
    {
        //! The CUDA/HIP vectors 1D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<aV2P::detail::CudaHipBuiltinTypes1, T>::value>>
        {
            using type = DimInt<1u>;
        };

        //! The CUDA/HIP vectors 2D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<aV2P::detail::CudaHipBuiltinTypes2, T>::value>>
        {
            using type = DimInt<2u>;
        };

        //! The CUDA/HIP vectors 3D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<aV2P::detail::CudaHipBuiltinTypes3, T>::value>>
        {
            using type = DimInt<3u>;
        };

        //! The CUDA/HIP vectors 4D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<aV2P::detail::CudaHipBuiltinTypes4, T>::value>>
        {
            using type = DimInt<4u>;
        };

        //! The CUDA/HIP vectors elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<aV2P::detail::isCudaHipBuiltInType<T>>>
        {
            using type = decltype(std::declval<T>().x);
        };

        template<typename TCudaHipBuiltin>
        struct GetExtents<TCudaHipBuiltin, std::enable_if_t<aV2P::detail::isCudaHipBuiltInType<TCudaHipBuiltin>>>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC auto operator()(TCudaHipBuiltin const& value) const
                -> Vec<Dim<TCudaHipBuiltin>, Idx<TCudaHipBuiltin>>
            {
                constexpr auto dim = Dim<TCudaHipBuiltin>::value;
                if constexpr(dim == 1)
                    return {value.x};
                else if constexpr(dim == 2)
                    return {value.y, value.x};
                else if constexpr(dim == 3)
                    return {value.z, value.y, value.x};
                else if constexpr(dim == 4)
                    return {value.w, value.z, value.y, value.x};
                else
                    static_assert(sizeof(value) == 0, "Not implemented");

                AV2P_UNREACHABLE({});
            }
        };

        template<typename TCudaHipBuiltin>
        struct GetOffsets<TCudaHipBuiltin, std::enable_if_t<aV2P::detail::isCudaHipBuiltInType<TCudaHipBuiltin>>>
            : GetExtents<TCudaHipBuiltin>
        {
        };

        //! The CUDA/HIP vectors idx type trait specialization.
        template<typename TIdx>
        struct IdxType<TIdx, std::enable_if_t<aV2P::detail::isCudaHipBuiltInType<TIdx>>>
        {
            using type = std::size_t;
        };
    } // namespace trait
} // namespace aV2P

#endif
