/* Copyright 2025 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>

// A functor that takes a single argument of type float and returns a value of type float
struct Duplicate
{
    template<typename T>
    AV2P_FN_HOST_ACC T operator()(T value) const
    {
        return value * 2;
    }
};

// A functor that takes two arguments (an accelerator object and a float) and returns a value of type float
struct Sin
{
    template<typename TAcc, typename T>
    AV2P_FN_HOST_ACC T operator()(TAcc const& acc, T value) const
    {
        return aV2P::math::sin(acc, value);
    }
};

using TestAccs1D = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, uint32_t>;

TEMPLATE_LIST_TEST_CASE("transform_pointers", "[algo]", TestAccs1D)
{
    using Acc = TestType;
    using Tag = aV2P::AccToTag<Acc>;
    using Platform = aV2P::Platform<Acc>;
    using Device = aV2P::Dev<Acc>;
    using Queue = aV2P::Queue<Device, aV2P::NonBlocking>;

    using Idx = aV2P::Idx<Acc>;
    // using Dim = aV2P::Dim<Acc>;
    // using Vec = aV2P::Vec<Dim, Idx>;

    const aV2P::PlatformCpu host_platform{};
    auto const host = aV2P::getDevByIdx(host_platform, 0u);

    const Platform platform{};
    const Device device = aV2P::getDevByIdx(platform, 0);
    Queue queue{device};

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    Idx size = 1000;

    // Allocate input and output buffers on the host.
    auto sample = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto result = aV2P::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffer with random numbers.
    for(Idx i = 0; i < size; ++i)
    {
        sample[i] = dist(rand);
    }

    // Allocate input and output buffers on the device.
    auto sample_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
    auto result_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);

    // Copy the input random numbers to the device.
    aV2P::memcpy(queue, sample_d, sample);

    // Apply the transform algorithm on the device.
    aV2P::transform<Tag>(queue, result_d.data(), result_d.data() + size, Duplicate{}, sample_d.data());

    // Copy the result to the host.
    aV2P::memcpy(queue, result, result_d);

    // Wait for all asynchronous operations to complete.
    aV2P::wait(queue);

    // Check the correctness of the results.
    for(Idx i = 0; i < size; ++i)
    {
        float expected = Duplicate{}(sample[i]);
        REQUIRE_THAT(result[i], Catch::Matchers::WithinULP(expected, 4));
    }
}

TEMPLATE_LIST_TEST_CASE("transform_buffer", "[algo]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Tag = aV2P::AccToTag<Acc>;
    using Platform = aV2P::Platform<Acc>;
    using Device = aV2P::Dev<Acc>;
    using Queue = aV2P::test::DefaultQueue<Device>;

    using Idx = aV2P::Idx<Acc>;
    using Dim = aV2P::Dim<Acc>;
    using Vec = aV2P::Vec<Dim, Idx>;

    const aV2P::PlatformCpu host_platform{};
    auto const host = aV2P::getDevByIdx(host_platform, 0u);

    const Platform platform{};
    const Device device = aV2P::getDevByIdx(platform, 0);
    Queue queue{device};

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    Vec size = aV2P::test::extentBuf<Dim, Idx>;

    // Allocate input and output buffers on the host.
    auto sample = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto result = aV2P::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffer with random numbers.
    for(Idx i = 0; i < size.prod(); ++i)
    {
        sample.data()[i] = dist(rand);
    }

    // Allocate input and output buffers on the device.
    auto sample_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
    auto result_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);

    // Copy the input random numbers to the device.
    aV2P::memcpy(queue, sample_d, sample);

    // Apply the transform algorithm on the device.
    aV2P::transform<Tag>(queue, result_d, Sin{}, sample_d);

    // Copy the result to the host.
    aV2P::memcpy(queue, result, result_d);

    // Wait for all asynchronous operations to complete.
    aV2P::wait(queue);

    // Check the correctness of the results.
    for(Idx i = 0; i < size.prod(); ++i)
    {
        float expected = std::sin(sample.data()[i]);
        REQUIRE_THAT(result.data()[i], Catch::Matchers::WithinULP(expected, 4));
    }
}
