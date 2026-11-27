/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

class VectorAddKernel
{
public:
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const a,
        alpaka::concepts::IMdSpan auto const b,
        alpaka::concepts::IMdSpan auto c,
        auto const& numElements) const -> void
    {
        using namespace alpaka;
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            numElements,
            [&](auto const&, auto&& simdA, auto&& simdB, auto&& simdC) constexpr
            { simdC = simdA.load() + simdB.load(); },
            a,
            b,
            c);
    }
};

auto run(auto const deviceSpec, auto const exec) -> int
{
    using namespace alpaka;
    using IdxVec = Vec<std::size_t, 1u>;
    using Data = std::uint32_t;

    IdxVec const extent(1024u);

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
    {
        std::cout << "SKIP: no device for " << deviceSpec.getApi().getName() << " "
                  << deviceSpec.getDeviceKind().getName() << std::endl;
        return 77;
    }

    onHost::Device device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    auto hostA = onHost::allocHost<Data>(extent);
    auto hostB = onHost::allocHostLike(hostA);
    auto hostC = onHost::allocHostLike(hostA);
    for(std::size_t i = 0u; i < extent.x(); ++i)
    {
        hostA[i] = static_cast<Data>(i);
        hostB[i] = static_cast<Data>(2u * i);
        hostC[i] = 0u;
    }

    auto accA = onHost::allocLike(device, hostA);
    auto accB = onHost::allocLike(device, hostB);
    auto accC = onHost::allocLike(device, hostC);

    std::uint32_t const elementsPerWorker = getNumElemPerThread<Data>(queue);
    Vec<std::size_t, 1u> chunkSize = 256u;
    auto frameSpec = onHost::FrameSpec{divCeil(extent, chunkSize * elementsPerWorker), chunkSize, exec};

    onHost::memcpy(queue, accA, hostA);
    onHost::memcpy(queue, accB, hostB);
    onHost::memset(queue, accC, std::uint8_t{0});
    queue.enqueue(frameSpec, KernelBundle{VectorAddKernel{}, accA, accB, accC, extent});
    onHost::memcpy(queue, hostC, accC);
    onHost::wait(queue);

    for(std::size_t i = 0u; i < extent.x(); ++i)
    {
        if(hostC[i] != hostA[i] + hostB[i])
        {
            std::cerr << "Wrong result at index " << i << ": " << hostC[i] << " != " << hostA[i] + hostB[i]
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "RESULT: " << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << " OK"
              << std::endl;
    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
#if defined(ALPAKA_TEST_CUDA)
    return run(onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu}, exec::gpuCuda);
#elif defined(ALPAKA_TEST_HIP)
    return run(onHost::DeviceSpec{api::hip, deviceKind::amdGpu}, exec::gpuHip);
#elif defined(ALPAKA_TEST_ONEAPI)
    return run(onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu}, exec::oneApi);
#else
    return run(onHost::DeviceSpec{api::host, deviceKind::cpu}, exec::cpuSerial);
#endif
}
