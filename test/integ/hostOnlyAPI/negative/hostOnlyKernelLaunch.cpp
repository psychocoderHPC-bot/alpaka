/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// Negative compile test: a host-only translation unit may construct and query a CUDA/HIP device
// and queue, but it must not be able to enqueue a kernel.  The native alpaka 3 launch guard turns
// that into one clear static_assert instead of a host call without device code.

#include <aV2P/alpaka.hpp>

#include <cstdint>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
using Acc = aV2P::AccGpuCudaRt<aV2P::DimInt<1u>, std::uint32_t>;
#else
using Acc = aV2P::AccGpuHipRt<aV2P::DimInt<1u>, std::uint32_t>;
#endif

struct Kernel
{
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const&) const -> void
    {
    }
};

int main()
{
    using Dev = aV2P::Dev<Acc>;
    using Queue = aV2P::Queue<Dev, aV2P::Blocking>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Vec = aV2P::Vec<Dim, Idx>;
    using WorkDiv = aV2P::WorkDivMembers<Dim, Idx>;

    auto const dev = aV2P::getDevByIdx(aV2P::Platform<Acc>{}, 0);
    Queue queue{dev};
    WorkDiv const workDiv{Vec{1u}, Vec{1u}, Vec{1u}};

    aV2P::exec<Acc>(queue, workDiv, Kernel{});
    return 0;
}
