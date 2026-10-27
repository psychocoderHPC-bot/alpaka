/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>

#include <cstdint>
#include <iostream>

// This example only makes sense with alpaka AccCpuOmp2Blocks backend enabled
// and OpenMP runtime supporting at least 3.0. Disable it for other cases.
#if defined _OPENMP && _OPENMP >= 200805 && AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED

//! OpenMP schedule demonstration kernel
//!
//! Prints distribution of alpaka thread indices between OpenMP threads.
//! Its operator() is reused in other kernels of this example.
//! Sets no schedule explicitly, so no schedule() clause is used.
struct OpenMPScheduleDefaultKernel
{
    template<typename TAcc>
    AV2P_FN_HOST auto operator()(TAcc const& acc) const -> void
    {
        // For simplicity assume 1d index space throughout this example
        using Idx = aV2P::Idx<TAcc>;
        Idx const globalThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0];

        // Print work distribution between threads for illustration
        printf(
            "alpaka global thread index %u is processed by OpenMP thread %d\n",
            static_cast<std::uint32_t>(globalThreadIdx),
            omp_get_thread_num());
    }
};

//! Kernel that sets the schedule via a static member.
//! We inherit OpenMPScheduleDefaultKernel just to reuse its operator().
struct OpenMPScheduleMemberKernel : public OpenMPScheduleDefaultKernel
{
    //! These two members are only checked for when the OmpSchedule trait is not specialized for this kernel type.

    //! OpenMP schedule kind to be used by the AccCpuOmp2Blocks accelerator,
    //! has to be static and constexpr.
    static constexpr auto ompScheduleKind = aV2P::omp::Schedule::Static;

    //! Member to set OpenMP chunk size, can be non-static and non-constexpr.
    //! Defining kind as member and not defining chunk will result in default chunk size for the kind.
    static constexpr int ompScheduleChunkSize = 1;
};

//! Kernel that sets the schedule via trait specialization.
//! We inherit OpenMPScheduleDefaultKernel just to reuse its operator().
//! The schedule trait specialization is given underneath this struct.
//! It has a higher priority than the internal static member.
struct OpenMPScheduleTraitKernel : public OpenMPScheduleDefaultKernel
{
};

namespace aV2P::trait
{
    //! Schedule trait specialization for OpenMPScheduleTraitKernel.
    //! This is the most general way to define a schedule.
    //! In case neither the trait nor the member are provided, there will be no schedule() clause.
    template<typename TAcc>
    struct OmpSchedule<OpenMPScheduleTraitKernel, TAcc>
    {
        template<typename TDim, typename... TArgs>
        AV2P_FN_HOST static auto getOmpSchedule(
            OpenMPScheduleTraitKernel const& /* kernelFnObj */,
            Vec<TDim, Idx<TAcc>> const& /* blockThreadExtent */,
            Vec<TDim, Idx<TAcc>> const& /* threadElemExtent */,
            TArgs const&... /* args */) -> aV2P::omp::Schedule
        {
            // Determine schedule at runtime for the given kernel and run parameters.
            // For this particular example kernel, TArgs is an empty pack and can be removed.
            return aV2P::omp::Schedule{aV2P::omp::Schedule::Dynamic, 2};
        }
    };
} // namespace aV2P::trait

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#    if defined(AV2P_CI) && !defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#    else
    using Idx = std::size_t;

    // OpenMP schedule illustrated by this example only has effect with
    // with the AccCpuOmp2Blocks accelerator.
    // This example also assumes 1d for simplicity.
    using Acc = aV2P::AccCpuOmp2Blocks<aV2P::DimInt<1>, Idx>;
    std::cout << "Using alpaka accelerator: " << aV2P::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = aV2P::Blocking;
    using Queue = aV2P::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    Idx const elementsPerGrid = 16u;
    Idx const elementsPerThread = 1u;

    OpenMPScheduleDefaultKernel openMPScheduleDefaultKernel;

    // Let alpaka calculate good block and grid sizes given our full problem extent
    aV2P::KernelCfg<Acc> kernelCfg = {elementsPerGrid, elementsPerThread};
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, devAcc, openMPScheduleDefaultKernel);

    // Run the kernel setting no schedule explicitly.
    std::cout << "OpenMPScheduleDefaultKernel setting no schedule explicitly:\n";
    aV2P::exec<Acc>(queue, workDiv, openMPScheduleDefaultKernel);
    aV2P::wait(queue);

    // Run the kernel setting the schedule via a trait
    std::cout << "\n\nOpenMPScheduleMemberKernel setting the schedule via a static member:\n";
    aV2P::exec<Acc>(queue, workDiv, OpenMPScheduleMemberKernel{});
    aV2P::wait(queue);

    // Run the kernel setting the schedule via a trait
    std::cout << "\n\nOpenMPScheduleTraitKernel setting the schedule via trait:\n";
    aV2P::exec<Acc>(queue, workDiv, OpenMPScheduleTraitKernel{});
    aV2P::wait(queue);

    return EXIT_SUCCESS;
#    endif
}
#else
auto main() -> int
{
    std::cout << "This example is disabled, as it requires OpenMP runtime version >= 3.0 and alpaka accelerator"
              << " AccCpuOmp2Blocks\n";
    return EXIT_SUCCESS;
}
#endif
