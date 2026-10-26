/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccGpuUniformCudaHipRt.hpp"
#include "aV2P/core/Assert.hpp"
#include "aV2P/core/Cuda.hpp"
#include "aV2P/core/Hip.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/exec/UniformElements.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/queue/QueueUniformCudaHipRtBlocking.hpp"
#include "aV2P/queue/QueueUniformCudaHipRtNonBlocking.hpp"
#include "aV2P/queue/Traits.hpp"
#include "aV2P/wait/Traits.hpp"
#include "aV2P/workdiv/WorkDivMembers.hpp"

#include <iostream>
#include <type_traits>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    template<typename TApi>
    class DevUniformCudaHipRt;

    namespace detail
    {
        template<typename TElem, typename TExtent, typename TPitchBytes>
        struct FillKernelND
        {
            template<typename TAcc>
            AV2P_FN_ACC void operator()(
                TAcc const& acc,
                TElem* ptr,
                TElem value,
                TExtent extent,
                TPitchBytes pitchBytes) const
            {
                for(auto const& idx : aV2P::uniformElementsND(acc, extent))
                {
                    // The host code checks that the pitches are a multiple of TElem's alignment.
                    std::uintptr_t offsetBytes = static_cast<std::uintptr_t>((pitchBytes * idx).sum());
                    TElem* elem = reinterpret_cast<TElem*>(
                        __builtin_assume_aligned(reinterpret_cast<std::uint8_t*>(ptr) + offsetBytes, alignof(TElem)));

                    // Write value at element address
                    *elem = value;
                }
            }
        };

        template<typename TElem>
        struct FillKernel0D
        {
            template<typename TAcc>
            AV2P_FN_ACC void operator()([[maybe_unused]] TAcc const& acc, TElem* ptr, TElem value) const
            {
                // A zero-dimensional buffer always has a single element.
                *ptr = value;
            }
        };


    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TApi>
        struct CreateTaskFill<TDim, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent, typename TViewFwd, typename TValue>
            AV2P_FN_HOST static auto createTaskFill(TViewFwd&& view, TValue const& value, TExtent const& extent)
            {
                using View = std::remove_reference_t<TViewFwd>;
                using Idx = aV2P::Idx<View>;
                using Acc = AccGpuUniformCudaHipRt<TApi, TDim, Idx>;
                using WorkDiv = aV2P::WorkDivMembers<TDim, Idx>;
                using Vec = aV2P::Vec<TDim, Idx>;
                using Elem = aV2P::Elem<View>;
                static_assert(
                    std::is_trivially_copyable_v<Elem>,
                    "Only trivially copyable types are supported for fill");

                if constexpr(TDim::value == 0)
                {
                    // A zero-dimensional buffer always has a single element.
                    WorkDiv grid{Vec{}, Vec{}, Vec{}};
                    return aV2P::createTaskKernel<Acc>(
                        grid,
                        aV2P::detail::FillKernel0D<Elem>{},
                        std::data(view),
                        value);
                }
                else
                {
                    // TODO: compute an efficient work division.
                    Vec const elements = Vec::ones();
                    Vec threads = Vec::ones();
                    threads.x() = 64;
                    Vec const blocks = Vec::ones();
                    WorkDiv grid = WorkDiv(blocks, threads, elements);

                    // Check that the pitches are a multiple of Elem's alignment.
                    auto pitches = getPitchesInBytes(view);
                    for([[maybe_unused]] auto pitch : pitches)
                    {
                        AV2P_ASSERT(static_cast<std::size_t>(pitch) % alignof(Elem) == 0);
                    }
                    return aV2P::createTaskKernel<Acc>(
                        grid,
                        aV2P::detail::FillKernelND<Elem, TExtent, Vec>{},
                        std::data(view),
                        value,
                        extent,
                        pitches);
                }
            }
        };
    } // namespace trait
} // namespace aV2P

#endif
