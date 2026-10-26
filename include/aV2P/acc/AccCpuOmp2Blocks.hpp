/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci,
 * Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "aV2P/atomic/AtomicCpu.hpp"
#include "aV2P/atomic/AtomicHierarchy.hpp"
#include "aV2P/atomic/AtomicNoOp.hpp"
#include "aV2P/atomic/AtomicOmpBuiltIn.hpp"
#include "aV2P/block/shared/dyn/BlockSharedMemDynMember.hpp"
#include "aV2P/block/shared/st/BlockSharedMemStMember.hpp"
#include "aV2P/block/sync/BlockSyncNoOp.hpp"
#include "aV2P/core/DemangleTypeNames.hpp"
#include "aV2P/grid/GridSyncBarrierCpuOmp.hpp"
#include "aV2P/idx/bt/IdxBtZero.hpp"
#include "aV2P/idx/gb/IdxGbRef.hpp"
#include "aV2P/intrinsic/IntrinsicCpu.hpp"
#include "aV2P/math/MathStdLib.hpp"
#include "aV2P/mem/fence/MemFenceOmp2Blocks.hpp"
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

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

namespace aV2P
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, bool TCooperative, typename... TArgs>
    class TaskKernelCpuOmp2Blocks;

    //! The CPU OpenMP 2.0 block accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a CPU device.
    //! It uses OpenMP 2.0 to implement the grid block parallelism.
    //! The block idx is restricted to 1x1x1.
    template<typename TDim, typename TIdx>
    class AccCpuOmp2Blocks final
        : public WorkDivMembers<TDim, TIdx>
        , public gb::IdxGbRef<TDim, TIdx>
        , public bt::IdxBtZero<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicCpu, // grid atomics
              AtomicOmpBuiltIn, // block atomics
              AtomicNoOp> // thread atomics
        , public math::MathStdLib
        , public BlockSharedMemDynMember<>
        , public BlockSharedMemStMember<>
        , public BlockSyncNoOp
        , public GridSyncOmp
        , public IntrinsicCpu
        , public MemFenceOmp2Blocks
#    ifdef AV2P_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandStdLib
#    endif
        , public warp::WarpSingleThread
        , public interface::Implements<InterfaceAcc, AccCpuOmp2Blocks<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, bool TCooperative, typename... TArgs>
        friend class ::aV2P::TaskKernelCpuOmp2Blocks;

        AccCpuOmp2Blocks(AccCpuOmp2Blocks const&) = delete;
        AccCpuOmp2Blocks(AccCpuOmp2Blocks&&) = delete;
        auto operator=(AccCpuOmp2Blocks const&) -> AccCpuOmp2Blocks& = delete;
        auto operator=(AccCpuOmp2Blocks&&) -> AccCpuOmp2Blocks& = delete;

    private:
        template<typename TWorkDiv>
        AV2P_FN_HOST AccCpuOmp2Blocks(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMember<>(staticMemBegin(), staticMemCapacity())
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    private:
        // getIdx
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.
    };

    namespace trait
    {
        //! The CPU OpenMP 2.0 block accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = AccCpuOmp2Blocks<TDim, TIdx>;
        };

        //! The CPU OpenMP 2.0 block single thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccCpuOmp2Blocks<TDim, TIdx>> : std::true_type
        {
        };

        //! The CPU OpenMP 2.0 block multi thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccCpuOmp2Blocks<TDim, TIdx>> : std::false_type
        {
        };

        //! The CPU OpenMP 2.0 block accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            AV2P_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> aV2P::AccDevProps<TDim, TIdx>
            {
                return {// m_multiProcessorCount
                        aV2P::core::clipCast<TIdx>(omp_get_max_threads()),
                        // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::ones(),
                        // m_blockThreadCountMax
                        static_cast<TIdx>(1),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(AccCpuOmp2Blocks<TDim, TIdx>::staticAllocBytes()),
                        // m_globalMemSizeBytes
                        getMemBytes(dev),
                        // m_cooperativeLaunch
                        true};
            }
        };

        //! The CPU OpenMP 2.0 block accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuOmp2Blocks<TDim, TIdx>>
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
                    std::format("AccCpuOmp2Blocks<{},{}>", TDim::value, core::demangled<TIdx>);
#    else
                    "AccCpuOmp2Blocks<"s + std::to_string(TDim::value) + ","s + std::string(core::demangled<TIdx>)
                    + ">"s;
#    endif
                return accName;
#    if AV2P_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
            }
        };

        //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //! The CPU OpenMP 2.0 block accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU OpenMP 2.0 block accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuOmp2Blocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            AV2P_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                if(workDiv.m_blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error(
                        "The given work division is not valid for a single thread Acc: "
                        + getAccName<AccCpuOmp2Blocks<TDim, TIdx>>() + ". Threads per block should be 1!");
                }

                return TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, false, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU OpenMP 2.0 block accelerator execution cooperative task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskCooperativeKernel<AccCpuOmp2Blocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            AV2P_FN_HOST static auto createTaskCooperativeKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(workDiv);
                auto const maxBlocks = omp_get_max_threads();
                if(gridBlockExtent.prod() > static_cast<TIdx>(maxBlocks))
                {
                    throw std::runtime_error(
                        "The number of requested blocks is larger than maximuma of the device for OpenMP 2.0 blocks "
                        "accelerator. Requested: "
                        + std::to_string(gridBlockExtent.prod()) + ", maximum allowed: " + std::to_string(maxBlocks)
                        + ". Use getMaxActiveBlocks().");
                }

                return TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, true, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU OpenMP 2.0 block execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PlatformType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = PlatformCpu;
        };

        //! The CPU OpenMP 2.0 block accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = aV2P::TagCpuOmp2Blocks;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuOmp2Blocks, TDim, TIdx>
        {
            using type = aV2P::AccCpuOmp2Blocks<TDim, TIdx>;
        };
    } // namespace trait
} // namespace aV2P

#endif
