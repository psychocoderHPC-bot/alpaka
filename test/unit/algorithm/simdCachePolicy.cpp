/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Reads both inputs and writes their sum using the default cache policy. */
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

/** Reads the left input with the default policy and the right input with the streaming policy. */
struct AddTwoInputsMixed
{
    constexpr void operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto out,
        concepts::SimdPtr auto const& lhs,
        concepts::SimdPtr auto const& rhs) const
    {
        out = lhs.load() + rhs.load(onAcc::simdCachePolicy::Stream{});
    }
};

/** Reads both inputs with the streaming policy. */
struct AddTwoInputsStream
{
    constexpr void operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto out,
        concepts::SimdPtr auto const& lhs,
        concepts::SimdPtr auto const& rhs) const
    {
        out = lhs.load(onAcc::simdCachePolicy::Stream{}) + rhs.load(onAcc::simdCachePolicy::Stream{});
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

/** Stencil functor for transformReduce, it receives SimdPtr and selects the policy at the load site. */
struct AddOneSimdStream
{
    constexpr auto operator()(concepts::SimdPtr auto const& value) const
    {
        using ValueType = typename std::remove_cvref_t<decltype(value)>::value_type;
        return value.load(onAcc::simdCachePolicy::Stream{}) + ValueType{1};
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
        auto const& extentMd) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::WorkerGroup{extentMd.fill(0), extentMd.fill(1)}};

        // (a) The same call site selects Default, mixed and Stream loads, the data sources are plain buffers.
        simdGrid.template concurrent<16u>(acc, extentMd, AddTwoInputs{}, outDefault, lhs, rhs);
        simdGrid.template concurrent<16u>(acc, extentMd, AddTwoInputsMixed{}, outMixed, lhs, rhs);
        simdGrid.template concurrent<16u>(acc, extentMd, AddTwoInputsStream{}, outStream, lhs, rhs);

        // (b) The policy is selected at the load site of a stencil functor, Default is the reference.
        using ReduceValueType = typename ALPAKA_TYPEOF(reduceDefault)::value_type;
        reduceDefault[0u]
            = simdGrid
                  .template transformReduce<16u>(acc, extentMd, ReduceValueType{0}, std::plus{}, AddOneSimd{}, lhs);
        reduceStream[0u] = simdGrid.template transformReduce<16u>(
            acc,
            extentMd,
            ReduceValueType{0},
            std::plus{},
            StencilFunc{AddOneSimdStream{}},
            lhs);
    }
};

namespace
{
    /** Minimal data source used to test the load customisation point without a memory view. */
    struct LegacyLoadSource
    {
        using value_type = int;
        using index_type = uint32_t;

        static consteval uint32_t dim()
        {
            return 1u;
        }

        static constexpr auto getAlignment()
        {
            return Alignment<16>{};
        }

        ALPAKA_FN_HOST_ACC constexpr value_type const& operator[](concepts::Vector auto const& idx) const
        {
            return m_data[idx[0u]];
        }

        ALPAKA_FN_HOST_ACC constexpr value_type& operator[](concepts::Vector auto const& idx)
        {
            return m_data[idx[0u]];
        }

        int* m_data = nullptr;
    };

    /** Data source whose load specialisation is policy-aware. */
    struct PolicyAwareLoadSource
    {
        using value_type = int;
        using index_type = uint32_t;

        static consteval uint32_t dim()
        {
            return 1u;
        }

        static constexpr auto getAlignment()
        {
            return Alignment<16>{};
        }

        ALPAKA_FN_HOST_ACC constexpr value_type const& operator[](concepts::Vector auto const& idx) const
        {
            return m_data[idx[0u]];
        }

        ALPAKA_FN_HOST_ACC constexpr value_type& operator[](concepts::Vector auto const& idx)
        {
            return m_data[idx[0u]];
        }

        int* m_data = nullptr;
    };
} // namespace

namespace alpaka::internal
{
    /** A third-party data source written against the policy-free load form keeps working. */
    template<alpaka::concepts::Alignment T_Alignment, alpaka::concepts::Vector T_Idx>
    struct LoadAsSimd::Op<LegacyLoadSource, T_Alignment, T_Idx>
    {
        template<uint32_t T_simdWidth>
        constexpr auto load(auto&& dataSource, T_Alignment, T_Idx const& idx) const
        {
            static_assert(
                std::is_same_v<LegacyLoadSource, ALPAKA_TYPEOF(dataSource)>,
                "Data source type must match the class template signature.");
            return Simd<int, T_simdWidth>([&](auto i) -> int { return dataSource.m_data[idx[0u] + i]; });
        }
    };

    /** A policy-aware data source sees the policy tag; the policy-free form stays as a fallback. */
    template<alpaka::concepts::Alignment T_Alignment, alpaka::concepts::Vector T_Idx>
    struct LoadAsSimd::Op<PolicyAwareLoadSource, T_Alignment, T_Idx>
    {
        template<uint32_t T_simdWidth>
        constexpr auto load(auto&& dataSource, T_Alignment, T_Idx const& idx) const
        {
            return load<T_simdWidth>(
                ALPAKA_FORWARD(dataSource),
                T_Alignment{},
                idx,
                onAcc::simdCachePolicy::Default{});
        }

        template<uint32_t T_simdWidth, typename T_Policy>
        constexpr auto load(auto&& dataSource, T_Alignment, T_Idx const& idx, T_Policy) const
        {
            static_assert(
                std::is_same_v<PolicyAwareLoadSource, ALPAKA_TYPEOF(dataSource)>,
                "Data source type must match the class template signature.");
            // The policy is a compile-time tag, the value proves that the specialisation received it.
            constexpr int offset = std::is_same_v<T_Policy, onAcc::simdCachePolicy::Stream> ? 100 : 0;
            return Simd<int, T_simdWidth>([&](auto i) -> int { return dataSource.m_data[idx[0u] + i] + offset; });
        }
    };
} // namespace alpaka::internal

/** The policy is passed at the load site and the load trait stays compatible with the policy-free form. */
TEST_CASE("SimdPtr load cache policy", "[simdCachePolicy]")
{
    constexpr uint32_t numElements = 32u;
    std::array<int, numElements> values{};
    for(uint32_t i = 0u; i < numElements; ++i)
        values[i] = static_cast<int>(i);

    using Idx = Vec<uint32_t, 1u>;
    using Pack = CVec<uint32_t, 4u>;

    // A data source which only implements the policy-free load form: the policy is ignored, both results match.
    LegacyLoadSource legacy{values.data()};
    auto legacyPtr = SimdPtr{legacy, Idx{0u}, Alignment<16>{}, Pack{}};
    auto legacyDefault = legacyPtr.load();
    auto legacyStream = legacyPtr.load(onAcc::simdCachePolicy::Stream{});
    for(uint32_t i = 0u; i < Pack{}.back(); ++i)
    {
        CHECK(static_cast<int>(legacyDefault[i]) == legacy.m_data[i]);
        CHECK(static_cast<int>(legacyStream[i]) == legacy.m_data[i]);
    }

    // A policy-aware specialisation sees the tag of the selected load.
    PolicyAwareLoadSource aware{values.data()};
    auto awarePtr = SimdPtr{aware, Idx{0u}, Alignment<16>{}, Pack{}};
    auto awareDefault = awarePtr.load();
    auto awareStream = awarePtr.load(onAcc::simdCachePolicy::Stream{});
    for(uint32_t i = 0u; i < Pack{}.back(); ++i)
    {
        CHECK(static_cast<int>(awareDefault[i]) == aware.m_data[i]);
        CHECK(static_cast<int>(awareStream[i]) == aware.m_data[i] + 100);
    }

    // A generator has no memory to hint, the default forwarding swallows the policy and the values are identical.
    auto generator = LinearizedIdxGenerator{Vec{numElements}};
    auto generatorPtr = SimdPtr{generator, Idx{0u}, Alignment<alignof(uint32_t)>{}, Pack{}};
    auto generatorDefault = generatorPtr.load();
    auto generatorStream = generatorPtr.load(onAcc::simdCachePolicy::Stream{});
    for(uint32_t i = 0u; i < Pack{}.back(); ++i)
    {
        CHECK(static_cast<uint32_t>(generatorDefault[i]) == i);
        CHECK(static_cast<uint32_t>(generatorStream[i]) == i);
    }

    // A shifted head is not 16 byte aligned, so Stream falls back to the plain load and both results match the
    // reference.
    auto misalignedPtr = legacyPtr[Idx{1u}];
    static_assert(decltype(misalignedPtr)::getAlignment().template get<int>() == alignof(int));
    auto misalignedDefault = misalignedPtr.load();
    auto misalignedStream = misalignedPtr.load(onAcc::simdCachePolicy::Stream{});
    for(uint32_t i = 0u; i < Pack{}.back(); ++i)
    {
        CHECK(static_cast<int>(misalignedDefault[i]) == legacy.m_data[1u + i]);
        CHECK(static_cast<int>(misalignedStream[i]) == legacy.m_data[1u + i]);
    }
}

/** Load one SIMD pack with the Stream policy and compare it against the values in the buffer.
 *
 * The pack width is a template argument, so this forces the wide chunk paths on a host build whose native width is
 * larger than one hinted load. 64 byte is a cache line and satisfies every hinted load instruction.
 */
template<typename DataType, uint32_t T_width>
void checkStreamLoadWidth()
{
    constexpr uint32_t alignment = 64u;

    struct alignas(alignment) AlignedBuffer
    {
        DataType data[T_width];
    };

    AlignedBuffer src{};
    for(uint32_t i = 0u; i < T_width; ++i)
        src.data[i] = static_cast<DataType>(i + 1);

    Simd<DataType, T_width> hinted{};
    hinted.copyFrom(src.data, Alignment<alignment>{}, onAcc::simdCachePolicy::Stream{});
    for(uint32_t i = 0u; i < T_width; ++i)
        CHECK(hinted[i] == src.data[i]);

    // A pointer which is only aligned for the element type must keep the plain load and produce the same values.
    Simd<DataType, T_width> fallback{};
    fallback.copyFrom(src.data, Alignment<alignof(DataType)>{}, onAcc::simdCachePolicy::Stream{});
    for(uint32_t i = 0u; i < T_width; ++i)
        CHECK(fallback[i] == src.data[i]);
}

TEST_CASE("Simd load cache policy derives width and alignment", "[simdCachePolicy]")
{
    // 8, 16, 32 and 64 byte packs; the 8 byte pack is narrower than every hinted load instruction.
    checkStreamLoadWidth<float, 2u>();
    checkStreamLoadWidth<float, 4u>();
    checkStreamLoadWidth<float, 8u>();
    checkStreamLoadWidth<float, 16u>();
    checkStreamLoadWidth<double, 2u>();
    checkStreamLoadWidth<double, 4u>();
    checkStreamLoadWidth<int, 4u>();
}

TEMPLATE_LIST_TEST_CASE("SimdAlgo cache policy at the load site", "", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue computeQueue = computeDev.makeQueue();

    // int, float and double cover the intrinsic-width packs of every backend, the chunked wide host packs and the
    // scalar remainder path for the types used in practice.
    auto runForDataType = [&]<typename DataType>()
    {
        auto const extentMd = Vec{1024u};
        onHost::SharedBuffer computeBufferLhs = onHost::alloc<DataType>(computeDev, extentMd);
        onHost::SharedBuffer computeBufferRhs = onHost::alloc<DataType>(computeDev, extentMd);

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
