/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace aV2P::adapter
{
    /** Legacy accelerator tag to alpaka 3 Api/DeviceKind/Executor mapping.
     *
     * This is the single place where legacy accelerator identities are translated.  The public
     * headers only alias the resulting `AccAlpaka3` instantiation.
     *
     * | legacy tag              | api               | device kind          | executor              | implemented |
     * |-------------------------|-------------------|----------------------|-----------------------|-------------|
     * | TagCpuSerial            | api::Host         | deviceKind::Cpu      | exec::CpuSerial       | yes         |
     * | TagCpuOmp2Blocks        | api::Host         | deviceKind::Cpu      | exec::CpuOmpBlocks    | yes         |
     * | TagCpuTbbBlocks         | api::Host         | deviceKind::Cpu      | exec::CpuTbbBlocks    | yes         |
     * | TagGpuCudaRt            | api::Cuda         | deviceKind::NvidiaGpu| exec::GpuCuda         | yes         |
     * | TagGpuHipRt             | api::Hip          | deviceKind::AmdGpu   | exec::GpuHip          | yes         |
     * | TagCpuSycl              | api::OneApi       | deviceKind::Cpu      | exec::OneApi          | yes         |
     * | TagGpuSyclIntel         | api::OneApi       | deviceKind::IntelGpu | exec::OneApi          | yes         |
     *
     * `TagCpuOmp2Threads` and `TagCpuThreads` deliberately have no mapping: mapping them to
     * `CpuSerial` or `CpuOmpBlocks` would silently change their semantics.
     */
    template<typename TTag>
    struct LegacyAccMapping;

    template<>
    struct LegacyAccMapping<TagCpuSerial>
    {
        using Api = alpaka::api::Host;
        using DeviceKind = alpaka::deviceKind::Cpu;
        using Executor = alpaka::exec::CpuSerial;
        static constexpr bool implemented = true;
    };

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    template<>
    struct LegacyAccMapping<TagCpuOmp2Blocks>
    {
        using Api = alpaka::api::Host;
        using DeviceKind = alpaka::deviceKind::Cpu;
        using Executor = alpaka::exec::CpuOmpBlocks;
        static constexpr bool implemented = true;
    };
#endif

#ifdef AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED
    template<>
    struct LegacyAccMapping<TagCpuTbbBlocks>
    {
        using Api = alpaka::api::Host;
        using DeviceKind = alpaka::deviceKind::Cpu;
        using Executor = alpaka::exec::CpuTbbBlocks;
        static constexpr bool implemented = true;
    };
#endif

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    template<>
    struct LegacyAccMapping<TagGpuCudaRt>
    {
        using Api = alpaka::api::Cuda;
        using DeviceKind = alpaka::deviceKind::NvidiaGpu;
        using Executor = alpaka::exec::GpuCuda;
        static constexpr bool implemented = true;
    };
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    template<>
    struct LegacyAccMapping<TagGpuHipRt>
    {
        using Api = alpaka::api::Hip;
        using DeviceKind = alpaka::deviceKind::AmdGpu;
        using Executor = alpaka::exec::GpuHip;
        static constexpr bool implemented = true;
    };
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    template<>
    struct LegacyAccMapping<TagCpuSycl>
    {
        using Api = alpaka::api::OneApi;
        using DeviceKind = alpaka::deviceKind::Cpu;
        using Executor = alpaka::exec::OneApi;
        static constexpr bool implemented = true;
    };

    template<>
    struct LegacyAccMapping<TagGpuSyclIntel>
    {
        using Api = alpaka::api::OneApi;
        using DeviceKind = alpaka::deviceKind::IntelGpu;
        using Executor = alpaka::exec::OneApi;
        static constexpr bool implemented = true;
    };
#endif

    //! Executors reached by an explicitly mapped legacy accelerator.
    //!
    //! Omp2Threads/Threads are deliberately absent: there is no alpaka 3 executor with their
    //! semantics, so they keep the `UnimplementedAcc` diagnostic instead of silently running on
    //! an unrelated executor.
    template<typename TExecutor>
    inline constexpr bool isImplementedExecutor
        = std::is_same_v<TExecutor, alpaka::exec::CpuSerial> || std::is_same_v<TExecutor, alpaka::exec::CpuOmpBlocks>
          || std::is_same_v<TExecutor, alpaka::exec::CpuTbbBlocks> || std::is_same_v<TExecutor, alpaka::exec::GpuCuda>
          || std::is_same_v<TExecutor, alpaka::exec::GpuHip> || std::is_same_v<TExecutor, alpaka::exec::OneApi>;

    template<typename TExecutor>
    inline constexpr bool isHostExecutor
        = std::is_same_v<TExecutor, alpaka::exec::CpuSerial> || std::is_same_v<TExecutor, alpaka::exec::CpuOmpBlocks>
          || std::is_same_v<TExecutor, alpaka::exec::CpuTbbBlocks>;
} // namespace aV2P::adapter
