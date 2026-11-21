/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Functor for onAcc::SimdAlgo::concurrent, the cache policy is forwarded to the SimdPtr loads. */
struct AddOne
{
    constexpr void operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto out,
        concepts::SimdPtr auto const& in) const
    {
        out = in.load() + 1;
    }
};

struct AddOneSimd
{
    constexpr auto operator()(concepts::Simd auto const& value) const
    {
        return value + 1;
    }
};

struct CachePolicyKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto outDefault,
        alpaka::concepts::IMdSpan auto outStream,
        alpaka::concepts::IMdSpan auto reduceDefault,
        alpaka::concepts::IMdSpan auto reduceStream,
        auto const& in,
        auto const& extentMd) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::WorkerGroup{extentMd.fill(0), extentMd.fill(1)}};

        simdGrid.template concurrent<16u, AutoAligned, onAcc::simdCachePolicy::Default>(
            acc,
            extentMd,
            AddOne{},
            outDefault,
            in);
        simdGrid.template concurrent<16u, AutoAligned, onAcc::simdCachePolicy::Stream>(
            acc,
            extentMd,
            AddOne{},
            outStream,
            in);

        reduceDefault[0u] = simdGrid.template transformReduce<16u, AutoAligned, onAcc::simdCachePolicy::Default>(
            acc,
            extentMd,
            0,
            std::plus{},
            AddOneSimd{},
            in);
        reduceStream[0u] = simdGrid.template transformReduce<16u, AutoAligned, onAcc::simdCachePolicy::Stream>(
            acc,
            extentMd,
            0,
            std::plus{},
            AddOneSimd{},
            in);
    }
};

TEMPLATE_LIST_TEST_CASE("SimdAlgo cache policy", "", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue computeQueue = computeDev.makeQueue();

    using DataType = int;
    auto const extentMd = Vec{1024};

    onHost::SharedBuffer computeBufferIn = onHost::alloc<DataType>(computeDev, extentMd);
    onHost::SharedBuffer computeBufferOutDefault = onHost::allocLike(computeDev, computeBufferIn);
    onHost::SharedBuffer computeBufferOutStream = onHost::allocLike(computeDev, computeBufferIn);
    onHost::SharedBuffer computeBufferReduceDefault = onHost::alloc<DataType>(computeDev, 1u);
    onHost::SharedBuffer computeBufferReduceStream = onHost::alloc<DataType>(computeDev, 1u);
    onHost::SharedBuffer hostBufferIn = onHost::allocHostLike(computeBufferIn);
    onHost::SharedBuffer hostBufferOutDefault = onHost::allocHostLike(computeBufferIn);
    onHost::SharedBuffer hostBufferOutStream = onHost::allocHostLike(computeBufferIn);
    onHost::SharedBuffer hostBufferReduceDefault = onHost::allocHostLike(computeBufferReduceDefault);
    onHost::SharedBuffer hostBufferReduceStream = onHost::allocHostLike(computeBufferReduceDefault);

    DataType iotaCounter = 0;
    for(auto& value : hostBufferIn)
    {
        value = iotaCounter;
        ++iotaCounter;
    }
    onHost::memcpy(computeQueue, computeBufferIn, hostBufferIn);

    computeQueue.enqueue(
        onHost::FrameSpec{1u, 1u, exec},
        KernelBundle{
            CachePolicyKernel{},
            computeBufferOutDefault,
            computeBufferOutStream,
            computeBufferReduceDefault,
            computeBufferReduceStream,
            computeBufferIn,
            extentMd});

    onHost::memcpy(computeQueue, hostBufferOutDefault, computeBufferOutDefault);
    onHost::memcpy(computeQueue, hostBufferOutStream, computeBufferOutStream);
    onHost::memcpy(computeQueue, hostBufferReduceDefault, computeBufferReduceDefault);
    onHost::memcpy(computeQueue, hostBufferReduceStream, computeBufferReduceStream);
    onHost::wait(computeQueue);

    DataType reference = 0;
    for(DataType idx = 0; idx < static_cast<DataType>(extentMd.product()); ++idx)
    {
        CHECK(hostBufferOutDefault[idx] == idx + 1);
        CHECK(hostBufferOutStream[idx] == hostBufferOutDefault[idx]);
        reference += idx + 1;
    }
    CHECK(*hostBufferReduceDefault.data() == reference);
    CHECK(*hostBufferReduceStream.data() == *hostBufferReduceDefault.data());
}
