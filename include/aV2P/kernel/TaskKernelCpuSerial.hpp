/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Specialized traits.
#include "aV2P/acc/Traits.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/platform/Traits.hpp"

// Implementation details.
#include "aV2P/acc/AccCpuSerial.hpp"
#include "aV2P/core/Decay.hpp"
#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/kernel/KernelFunctionAttributes.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/meta/NdLoop.hpp"
#include "aV2P/platform/PlatformCpu.hpp"
#include "aV2P/workdiv/WorkDivMembers.hpp"

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#if AV2P_DEBUG >= AV2P_DEBUG_MINIMAL
#    include <iostream>
#endif

#ifdef AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace aV2P
{
    //! The CPU serial execution task implementation.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuSerial final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        template<typename TWorkDiv>
        AV2P_FN_HOST TaskKernelCpuSerial(TWorkDiv&& workDiv, TKernelFnObj kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(std::move(kernelFnObj))
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //! Executes the kernel function object.
        AV2P_FN_HOST auto operator()() const -> void
        {
            AV2P_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(*this);
            auto const blockThreadExtent = getWorkDiv<Block, Threads>(*this);
            auto const threadElemExtent = getWorkDiv<Thread, Elems>(*this);

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes = std::apply(
                [&](std::decay_t<TArgs> const&... args)
                {
                    return getBlockSharedMemDynSizeBytes<AccCpuSerial<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args);

#    if AV2P_DEBUG >= AV2P_DEBUG_FULL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif

            AccCpuSerial<TDim, TIdx> acc(
                *static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
                blockSharedMemDynSizeBytes);

            // Execute the blocks serially.
            meta::ndLoopIncIdx(
                gridBlockExtent,
                [&](Vec<TDim, TIdx> const& blockThreadIdx)
                {
                    acc.m_gridBlockIdx = blockThreadIdx;

                    std::apply(
                        [&](auto&&... argsWithAcc) {
                            detail::checkKernelReturnType(
                                m_kernelFnObj,
                                std::forward<decltype(argsWithAcc)>(argsWithAcc)...);
                        },
                        std::tuple_cat(std::tie(acc), m_args));
                    std::apply(m_kernelFnObj, std::tuple_cat(std::tie(acc), m_args));

                    // After a block has been processed, the shared memory has to be deleted.
                    freeSharedVars(acc);
                });
        }

    private:
        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace trait
    {
        //! The CPU serial execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccCpuSerial<TDim, TIdx>;
        };

        //! The CPU serial execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevCpu;
        };

        //! The CPU serial execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The CPU serial execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PlatformType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PlatformCpu;
        };

        //! The CPU serial execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };

        //! \brief Specialisation of the class template FunctionAttributes
        //! \tparam TDev The device type.
        //! \tparam TDim The dimensionality of the accelerator device properties.
        //! \tparam TIdx The idx type of the accelerator device properties.
        //! \tparam TKernelFn Kernel function object type.
        //! \tparam TArgs Kernel function object argument types as a parameter pack.
        template<typename TDev, typename TDim, typename TIdx, typename TKernelFn, typename... TArgs>
        struct FunctionAttributes<AccCpuSerial<TDim, TIdx>, TDev, TKernelFn, TArgs...>
        {
            //! \param dev The device instance
            //! \param kernelFn The kernel function object which should be executed.
            //! \param args The kernel invocation arguments.
            //! \return KernelFunctionAttributes instance. The default version always returns an instance with zero
            //! fields. For CPU, the field of max threads allowed by kernel function for the block is 1.
            AV2P_FN_HOST static auto getFunctionAttributes(
                TDev const& dev,
                [[maybe_unused]] TKernelFn const& kernelFn,
                [[maybe_unused]] TArgs&&... args) -> aV2P::KernelFunctionAttributes
            {
                aV2P::KernelFunctionAttributes kernelFunctionAttributes;

                // set function properties for maxThreadsPerBlock to device properties, since API doesn't have function
                // properties function.
                auto const& props = aV2P::getAccDevProps<AccCpuSerial<TDim, TIdx>>(dev);
                kernelFunctionAttributes.maxThreadsPerBlock = static_cast<int>(props.m_blockThreadCountMax);
                kernelFunctionAttributes.maxDynamicSharedSizeBytes
                    = static_cast<int>(aV2P::BlockSharedDynMemberAllocKiB * 1024);
                return kernelFunctionAttributes;
            }
        };

        //! The CPU serial get max active blocks for cooperative kernel specialization.
        template<typename TDev, typename TKernelFnObj, typename TDim, typename TIdx, typename... TArgs>
        struct MaxActiveBlocks<AccCpuSerial<TDim, TIdx>, TDev, TKernelFnObj, TDim, TIdx, TArgs...>
        {
            AV2P_FN_HOST static auto getMaxActiveBlocks(
                TKernelFnObj const& /*kernelFnObj*/,
                TDev const& /*device*/,
                aV2P::Vec<TDim, TIdx> const& /*blockThreadExtent*/,
                aV2P::Vec<TDim, TIdx> const& /*threadElemExtent*/,
                TArgs const&... /*args*/) -> int
            {
                return 1;
            }
        };
    } // namespace trait
} // namespace aV2P

#endif
