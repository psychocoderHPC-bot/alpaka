/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <exception>
#include <functional>
#include <string>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Reads both inputs and writes their sum, the cache policy is carried by each data source. */
struct AddTwoInputs
{
    constexpr void operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto out,
        concepts::SimdPtr auto const& lhs,
        concepts::SimdPtr auto const& rhs) const
    {
        out = lhs.load() + rhs.load();
    }
};

struct AddOneSimd
{
    constexpr auto operator()(concepts::Simd auto const& value) const
    {
        using ValueType = typename std::remove_cvref_t<decltype(value)>::value_type;
        return value + ValueType{1};
    }
};

struct CachePolicyKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto outDefault,
        alpaka::concepts::IMdSpan auto outMixed,
        alpaka::concepts::IMdSpan auto outStream,
        alpaka::concepts::IMdSpan auto reduceDefault,
        alpaka::concepts::IMdSpan auto reduceStream,
        auto const& lhs,
        auto const& rhs,
        auto const& lhsStream,
        auto const& rhsStream,
        auto const& extentMd) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::WorkerGroup{extentMd.fill(0), extentMd.fill(1)}};

        // (a) Default on one source and Stream on the other source within one concurrent call.
        simdGrid.template concurrent<16u>(acc, extentMd, AddTwoInputs{}, outMixed, lhs, rhsStream);
        // Reference results without any hint.
        simdGrid.template concurrent<16u>(acc, extentMd, AddTwoInputs{}, outDefault, lhs, rhs);
        // Both sources hinted, to compare against the Default result.
        simdGrid.template concurrent<16u>(acc, extentMd, AddTwoInputs{}, outStream, lhsStream, rhsStream);

        // (b) Stream through transformReduce, Default as reference.
        using ReduceValueType = typename ALPAKA_TYPEOF(reduceDefault)::value_type;
        reduceDefault[0u]
            = simdGrid
                  .template transformReduce<16u>(acc, extentMd, ReduceValueType{0}, std::plus{}, AddOneSimd{}, lhs);
        reduceStream[0u] = simdGrid.template transformReduce<16u>(
            acc,
            extentMd,
            ReduceValueType{0},
            std::plus{},
            AddOneSimd{},
            lhsStream);
    }
};

TEMPLATE_LIST_TEST_CASE("SimdAlgo per-source cache policy", "", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue computeQueue = computeDev.makeQueue();

    // int, float and double cover the 16 byte SIMD packs int4, float4 and double2, i.e. the CUDA and HIP 16 byte
    // load path, the host SSE path and the scalar remainder path for the types used in practice.
    auto runForDataType = [&]<typename DataType>()
    {
        // The policy is a property of the data source type, not of the algorithm call.
        auto const extentMd = Vec{1024u};
        onHost::SharedBuffer computeBufferLhs = onHost::alloc<DataType>(computeDev, extentMd);
        onHost::SharedBuffer computeBufferRhs = onHost::alloc<DataType>(computeDev, extentMd);
        auto streamLhsView = withCachePolicy(onAcc::simdCachePolicy::Stream{}, computeBufferLhs.getView());
        // The owning buffer overload derives the view internally.
        auto streamLhs = withCachePolicy(onAcc::simdCachePolicy::Stream{}, computeBufferLhs);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<decltype(streamLhsView)>);
        STATIC_REQUIRE(std::is_trivially_copyable_v<decltype(streamLhsView)>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<decltype(streamLhsView)>);
        STATIC_REQUIRE(std::same_as<alpaka::CachePolicyOf_t<decltype(streamLhsView)>, onAcc::simdCachePolicy::Stream>);
        STATIC_REQUIRE(
            std::same_as<alpaka::CachePolicyOf_t<decltype(computeBufferLhs)>, onAcc::simdCachePolicy::Default>);
        STATIC_REQUIRE(std::same_as<decltype(streamLhsView.getExtents()), decltype(computeBufferLhs.getExtents())>);
        STATIC_REQUIRE(
            std::same_as<decltype(streamLhsView.getAlignment()), decltype(computeBufferLhs.getAlignment())>);
        STATIC_REQUIRE(requires { streamLhsView[Vec{0u}]; });
        STATIC_REQUIRE(std::same_as<decltype(streamLhs), decltype(streamLhsView)>);

        // (d) 1024 is a multiple of the SIMD width, 1002 leaves a remainder for the scalar tail path.
        for(uint32_t numElements : {1024u, 1002u})
        {
            auto const runExtentMd = Vec{numElements};

            onHost::SharedBuffer hostLhs = onHost::allocHostLike(computeBufferLhs);
            onHost::SharedBuffer hostRhs = onHost::allocHostLike(computeBufferRhs);
            for(uint32_t i = 0; i < extentMd.product(); ++i)
            {
                hostLhs[i] = static_cast<DataType>(i);
                hostRhs[i] = static_cast<DataType>(3 * i + 1);
            }
            onHost::memcpy(computeQueue, computeBufferLhs, hostLhs);
            onHost::memcpy(computeQueue, computeBufferRhs, hostRhs);

            onHost::SharedBuffer computeBufferOutDefault = onHost::allocLike(computeDev, computeBufferLhs);
            onHost::SharedBuffer computeBufferOutMixed = onHost::allocLike(computeDev, computeBufferLhs);
            onHost::SharedBuffer computeBufferOutStream = onHost::allocLike(computeDev, computeBufferLhs);
            onHost::SharedBuffer computeBufferReduceDefault = onHost::alloc<DataType>(computeDev, 1u);
            onHost::SharedBuffer computeBufferReduceStream = onHost::alloc<DataType>(computeDev, 1u);

            onHost::SharedBuffer hostOutDefault = onHost::allocHostLike(computeBufferOutDefault);
            onHost::SharedBuffer hostOutMixed = onHost::allocHostLike(computeBufferOutMixed);
            onHost::SharedBuffer hostOutStream = onHost::allocHostLike(computeBufferOutStream);
            onHost::SharedBuffer hostReduceDefault = onHost::allocHostLike(computeBufferReduceDefault);
            onHost::SharedBuffer hostReduceStream = onHost::allocHostLike(computeBufferReduceStream);

            computeQueue.enqueue(
                onHost::FrameSpec{1u, 1u, exec},
                KernelBundle{
                    CachePolicyKernel{},
                    computeBufferOutDefault,
                    computeBufferOutMixed,
                    computeBufferOutStream,
                    computeBufferReduceDefault,
                    computeBufferReduceStream,
                    computeBufferLhs,
                    computeBufferRhs,
                    withCachePolicy(onAcc::simdCachePolicy::Stream{}, computeBufferLhs),
                    withCachePolicy(onAcc::simdCachePolicy::Stream{}, computeBufferRhs.getView()),
                    runExtentMd});

            onHost::memcpy(computeQueue, hostOutDefault, computeBufferOutDefault);
            onHost::memcpy(computeQueue, hostOutMixed, computeBufferOutMixed);
            onHost::memcpy(computeQueue, hostOutStream, computeBufferOutStream);
            onHost::memcpy(computeQueue, hostReduceDefault, computeBufferReduceDefault);
            onHost::memcpy(computeQueue, hostReduceStream, computeBufferReduceStream);
            onHost::wait(computeQueue);

            DataType referenceReduce = DataType{0};
            for(uint32_t i = 0; i < numElements; ++i)
            {
                DataType const expected = hostLhs[i] + hostRhs[i];
                // (c) every Stream variant must produce the same result as the Default variant.
                CHECK(hostOutDefault[i] == expected);
                CHECK(hostOutMixed[i] == expected);
                CHECK(hostOutStream[i] == expected);
                referenceReduce += hostLhs[i] + DataType{1};
            }
            CHECK(*hostReduceDefault.data() == referenceReduce);
            CHECK(*hostReduceStream.data() == referenceReduce);
        }
    };

    /** Execute the checks for one value type, skipping it if the device cannot run it.
     *
     * alpaka has no portable value-type capability query, devices without fp64 (for example some SYCL GPUs) reject
     * the kernel launch instead. Only that rejection is turned into a skip, so int and float still run there.
     */
    auto runOrSkipUnsupportedType = [&]<typename DataType>()
    {
        try
        {
            runForDataType.template operator()<DataType>();
        }
        catch(std::exception const& exception)
        {
            std::string const message = exception.what();
            if(message.find("fp64") == std::string::npos)
                throw;
            WARN("Skipping the fp64 instantiation: " << message);
        }
    };

    runOrSkipUnsupportedType.template operator()<int>();
    runOrSkipUnsupportedType.template operator()<float>();
    runOrSkipUnsupportedType.template operator()<double>();
}
