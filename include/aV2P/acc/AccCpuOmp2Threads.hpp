/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "aV2P/atomic/AtomicCpu.hpp"
#include "aV2P/atomic/AtomicHierarchy.hpp"
#include "aV2P/atomic/AtomicOmpBuiltIn.hpp"
#include "aV2P/block/shared/dyn/BlockSharedMemDynMember.hpp"
#include "aV2P/block/shared/st/BlockSharedMemStMemberMasterSync.hpp"
#include "aV2P/block/sync/BlockSyncBarrierOmp.hpp"
#include "aV2P/core/DemangleTypeNames.hpp"
#include "aV2P/grid/GridSyncBarrierCpuOmp.hpp"
#include "aV2P/idx/bt/IdxBtOmp.hpp"
#include "aV2P/idx/gb/IdxGbRef.hpp"
#include "aV2P/intrinsic/IntrinsicCpu.hpp"
#include "aV2P/math/MathStdLib.hpp"
#include "aV2P/mem/fence/MemFenceOmp2Threads.hpp"
#include "aV2P/rand/RandDefault.hpp"
#include "aV2P/rand/RandStdLib.hpp"
#include "aV2P/warp/WarpSingleThread.hpp"
#include "aV2P/workdiv/WorkDivMembers.hpp"

// Specialized traits.
#include "aV2P/acc/Traits.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/platform/Traits.hpp"

// Implementation details.
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/ClipCast.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/dev/DevCpu.hpp"

#ifdef __cpp_lib_format
#    include <format>
#endif
#include <limits>
#include <string>

#ifdef AV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#    if _OPENMP < 200203
#        error If AV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <omp.h>

namespace aV2P
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuOmp2Threads;

    //! The CPU OpenMP 2.0 thread accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a CPU device.
    //! It uses OpenMP 2.0 to implement the block thread parallelism.
    template<typename TDim, typename TIdx>
    class AccCpuOmp2Threads final
        : public WorkDivMembers<TDim, TIdx>
        , public gb::IdxGbRef<TDim, TIdx>
        , public bt::IdxBtOmp<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicCpu, // grid atomics
              AtomicOmpBuiltIn, // block atomics
              AtomicOmpBuiltIn> // thread atomics
        , public math::MathStdLib
        , public BlockSharedMemDynMember<>
        , public BlockSharedMemStMemberMasterSync<>
        , public BlockSyncBarrierOmp
        , public GridSyncOmp
        , public IntrinsicCpu
        , public MemFenceOmp2Threads
#    ifdef AV2P_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandStdLib
#    endif
        , public warp::WarpSingleThread
        , public interface::Implements<InterfaceAcc, AccCpuOmp2Threads<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::aV2P::TaskKernelCpuOmp2Threads;

        AccCpuOmp2Threads(AccCpuOmp2Threads const&) = delete;
        AccCpuOmp2Threads(AccCpuOmp2Threads&&) = delete;
        auto operator=(AccCpuOmp2Threads const&) -> AccCpuOmp2Threads& = delete;
        auto operator=(AccCpuOmp2Threads&&) -> AccCpuOmp2Threads& = delete;

    private:
        template<typename TWorkDiv>
        AV2P_FN_HOST AccCpuOmp2Threads(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMemberMasterSync<>(
                  staticMemBegin(),
                  staticMemCapacity(),
                  [this]() { syncBlockThreads(*this); },
                  []() noexcept { return (::omp_get_thread_num() == 0); })
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    private:
        // getIdx
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.
    };

    namespace trait
    {
        //! The CPU OpenMP 2.0 thread accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuOmp2Threads<TDim, TIdx>>
        {
            using type = AccCpuOmp2Threads<TDim, TIdx>;
        };

        //! The CPU OpenMP 2.0 thread single thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccCpuOmp2Threads<TDim, TIdx>> : std::false_type
        {
        };

        //! The CPU OpenMP 2.0 thread multi thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccCpuOmp2Threads<TDim, TIdx>> : std::true_type
        {
        };

        //! The CPU OpenMP 2.0 thread accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuOmp2Threads<TDim, TIdx>>
        {
            AV2P_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> aV2P::AccDevProps<TDim, TIdx>
            {
#    ifdef AV2P_CI
                auto const blockThreadCountMax = aV2P::core::clipCast<TIdx>(std::min(4, ::omp_get_max_threads()));
#    else
                auto const blockThreadCountMax = aV2P::core::clipCast<TIdx>(::omp_get_max_threads());
#    endif
                auto const memBytes = getMemBytes(dev);
                return {// m_multiProcessorCount
                        static_cast<TIdx>(1),
                        // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        memBytes,
                        // m_globalMemSizeBytes
                        memBytes,
                        // m_cooperativeLaunch
                        true};
            }
        };

        //! The CPU OpenMP 2.0 thread accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuOmp2Threads<TDim, TIdx>>
        {
            AV2P_FN_HOST static auto getAccName() -> std::string
            {
#    if AV2P_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wexit-time-destructors"
#    endif
                using namespace std::literals;
                static std::string const accName =
#    ifdef __cpp_lib_format
                    std::format("AccCpuOmp2Threads<{},{}>", TDim::value, core::demangled<TIdx>);
#    else
                    "AccCpuOmp2Threads<"s + std::to_string(TDim::value) + ","s + std::string(core::demangled<TIdx>)
                    + ">"s;
#    endif
                return accName;
#    if AV2P_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
            }
        };

        //! The CPU OpenMP 2.0 thread accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuOmp2Threads<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //! The CPU OpenMP 2.0 thread accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuOmp2Threads<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU OpenMP 2.0 thread accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuOmp2Threads<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            AV2P_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU OpenMP 2.0 thread accelerator execution cooperative task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskCooperativeKernel<AccCpuOmp2Threads<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            AV2P_FN_HOST static auto createTaskCooperativeKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(workDiv);
                if(gridBlockExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error("OpenMP 2.0 thread accelerator supports only a single block operation "
                                             "with cooperative kernel!\n"
                                             "Consider useing a different CPU accelerator.");
                }

                return TaskKernelCpuOmp2Threads<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU OpenMP 2.0 thread execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PlatformType<AccCpuOmp2Threads<TDim, TIdx>>
        {
            using type = PlatformCpu;
        };

        //! The CPU OpenMP 2.0 thread accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuOmp2Threads<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuOmp2Threads<TDim, TIdx>>
        {
            using type = aV2P::TagCpuOmp2Threads;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuOmp2Threads, TDim, TIdx>
        {
            using type = aV2P::AccCpuOmp2Threads<TDim, TIdx>;
        };
    } // namespace trait
} // namespace aV2P

#endif
