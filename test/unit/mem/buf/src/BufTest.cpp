/* Copyright 2025 Axel Huebl, Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Jan Stephan,
 *                Aurora Perego, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/mem/buf/Traits.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

namespace buftest
{
    template<typename TDim, typename TDev, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(TDev dev, TExtent extent)
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }

    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevCpu dev, TExtent extent) -> aV2P::BufCpu<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }

#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevCudaRt dev, TExtent extent) -> aV2P::BufCudaRt<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevHipRt dev, TExtent extent) -> aV2P::BufHipRt<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(AV2P_ACC_SYCL_ENABLED) and defined(AV2P_SYCL_ONEAPI_CPU)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevCpuSycl dev, TExtent extent) -> aV2P::BufCpuSycl<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(AV2P_ACC_SYCL_ENABLED) and defined(AV2P_SYCL_ONEAPI_GPU)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevGpuSyclIntel dev, TExtent extent) -> aV2P::BufGpuSyclIntel<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(AV2P_ACC_SYCL_ENABLED) and defined(AV2P_SYCL_ONEAPI_GPU_NVIDIA)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevGpuSyclNvidia dev, TExtent extent) -> aV2P::BufGpuSyclNvidia<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(AV2P_ACC_SYCL_ENABLED) and defined(AV2P_SYCL_ONEAPI_GPU_AMD)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevGpuSyclAmd dev, TExtent extent) -> aV2P::BufGpuSyclAmd<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_FPGA)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(aV2P::DevFpgaSyclIntel dev, TExtent extent) -> aV2P::BufFpgaSyclIntel<TElem, TDim, TIdx>
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif

} // namespace buftest

template<typename TAcc>
static auto testBufferMutable(aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent) -> void
{
    using Dev = aV2P::Dev<TAcc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    auto const platformAcc = aV2P::Platform<TAcc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);
    Queue queue(dev);

    // aV2P::malloc
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);

    auto const offset = aV2P::Vec<Dim, Idx>::zeros();
    aV2P::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    aV2P::test::testViewMutable<TAcc>(queue, buf);
}

template<typename TAcc>
static auto testAsyncBufferMutable(aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent) -> void
{
    using Dev = aV2P::Dev<TAcc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    auto const platformAcc = aV2P::Platform<TAcc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);
    Queue queue(dev);

    // memory is allocated when the queue reaches this point
    auto buf = aV2P::allocAsyncBuf<Elem, Idx>(queue, extent);

    // asynchronous operations can be submitted to the queue immediately
    aV2P::test::testViewMutable<TAcc>(queue, buf);

    // synchronous operations must wait for the memory to be available
    aV2P::wait(queue);
    auto const offset = aV2P::Vec<Dim, Idx>::zeros();
    aV2P::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    // the buffer will queue the deallocation of the memory when it goes out of scope,
    // and extend the lifetime of the queue until all memory operations have completed.
}

TEMPLATE_LIST_TEST_CASE("memBufBasicTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    testBufferMutable<Acc>(aV2P::test::extentBuf<Dim, Idx>);
}

TEMPLATE_LIST_TEST_CASE("memBufZeroSizeTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    auto const extent = aV2P::Vec<Dim, Idx>::zeros();

    testBufferMutable<Acc>(extent);
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncBasicTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    if constexpr(aV2P::hasAsyncBufSupport<aV2P::Dev<Acc>, Dim>)
    {
        testAsyncBufferMutable<Acc>(aV2P::test::extentBuf<Dim, Idx>);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.");
    }
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncZeroSizeTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    if constexpr(aV2P::hasAsyncBufSupport<aV2P::Dev<Acc>, Dim>)
    {
        auto const extent = aV2P::Vec<Dim, Idx>::zeros();
        testAsyncBufferMutable<Acc>(extent);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.");
    }
}

template<typename TAcc>
static auto testBufferImmutable(aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent) -> void
{
    using Elem = float;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    auto const platformAcc = aV2P::Platform<TAcc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    // aV2P::malloc
    auto const buf = aV2P::allocBuf<Elem, Idx>(dev, extent);

    auto const offset = aV2P::Vec<Dim, Idx>::zeros();
    aV2P::test::testViewImmutable<Elem>(buf, dev, extent, offset);
}

TEMPLATE_LIST_TEST_CASE("memBufConstTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    testBufferImmutable<Acc>(aV2P::test::extentBuf<Dim, Idx>);
}

template<typename TAcc>
static auto testAsyncBufferImmutable(aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent) -> void
{
    {
        using Dev = aV2P::Dev<TAcc>;
        using Queue = aV2P::test::DefaultQueue<Dev>;

        using Elem = float;
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;

        auto const platformAcc = aV2P::Platform<TAcc>{};
        auto const dev = aV2P::getDevByIdx(platformAcc, 0);
        Queue queue(dev);

        // memory is allocated when the queue reaches this point
        auto const buf = aV2P::allocAsyncBuf<Elem, Idx>(queue, extent);

        // synchronous operations must wait for the memory to be available
        aV2P::wait(queue);
        auto const offset = aV2P::Vec<Dim, Idx>::zeros();
        aV2P::test::testViewImmutable<Elem>(buf, dev, extent, offset);

        // The buffer will queue the deallocation of the memory when it goes out of scope,
        // and extend the lifetime of the queue until all memory operations have completed.
        // Delay the end of the queue to push the buffer deletetion task after
        // all local refs to queue have been dropped.
        aV2P::enqueue(queue, []() { std::this_thread::sleep_for(std::chrono::microseconds(1000)); });
        [](auto) {}(std::move(queue));
    }

    // Give the queue, including the buffer's deleter time to complete, we
    // cannot synchronize here because we dropped the handle to see how it
    // behaves when it self-destructs.
    std::this_thread::sleep_for(std::chrono::microseconds(1200));
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncConstTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    if constexpr(aV2P::hasAsyncBufSupport<aV2P::Dev<Acc>, Dim>)
    {
        testAsyncBufferImmutable<Acc>(aV2P::test::extentBuf<Dim, Idx>);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.");
    }
}

template<typename TAcc>
static auto testBufferAccessorAdaptor(
    aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent,
    aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& index) -> void
{
    using Elem = float;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    // assume dimensionality up to 4
    CHECK(Dim::value <= 4);

    auto const platformAcc = aV2P::Platform<TAcc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);

    // check that the array subscript operator access the correct element
    auto const& pitch = aV2P::getPitchesInBytes(buf);
    INFO("buffer extent: " << extent << " elements");
    INFO("buffer pitch: " << pitch << " bytes");
    CHECK((index < extent).all());

    auto const base = reinterpret_cast<uintptr_t>(std::data(buf));
    auto const expected = base + static_cast<uintptr_t>((pitch * index).sum());
    INFO("element " << index << " expected at offset " << expected - base);
    using Platform = aV2P::Platform<TAcc>;
    if constexpr(
        std::is_same_v<Platform, aV2P::PlatformCpu> or std::is_same_v<aV2P::AccToTag<TAcc>, aV2P::TagCpuSycl>)
    {
        INFO("element " << index << " returned at offset " << reinterpret_cast<uintptr_t>(&buf[index]) - base);

        CHECK(reinterpret_cast<Elem*>(expected) == &buf[index]);

        // check that an out-of-bound access is detected
        if constexpr(Dim::value > 0)
            CHECK_THROWS_AS((void) buf.at(extent), std::out_of_range);
    }
}

TEMPLATE_LIST_TEST_CASE("memBufAccessorAdaptorTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    testBufferAccessorAdaptor<Acc>(aV2P::test::extentBuf<Dim, Idx>, aV2P::test::offset<Dim, Idx>);
}

// aV2P 0D limitation: alpaka 3.0.0 has no zero-dimensional memory support
// (onHost::alloc rejects Vec<T,0> extents via dim()/x() in api/util.hpp:106;
// View requires T_Extents::UniVec, mem/View.hpp:104).
// Restore when alpaka 3 supports 0D Buf/View/alloc; re-enable together with the adapter.
#if 0
TEMPLATE_LIST_TEST_CASE("memBufMove", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Idx = aV2P::Idx<Acc>;
    using Elem = std::size_t;
    using DimExtent = aV2P::DimInt<0>;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);
    auto queue = aV2P::Queue<Acc, aV2P::Blocking>{dev};
    auto const extent = aV2P::Vec<DimExtent, Idx>{};

    auto write = [&](auto& buf, Elem value)
    {
        auto v = aV2P::createView(devHost, &value, extent);
        aV2P::memcpy(queue, buf, v);
    };
    auto read = [&](auto const& buf)
    {
        Elem value{};
        auto v = aV2P::createView(devHost, &value, extent);
        aV2P::memcpy(queue, v, buf);
        return value;
    };

    // move constructor
    {
        auto buf1 = buftest::allocBuf<DimExtent, Elem, Idx>(dev, extent);
        write(buf1, 1);
        auto buf2{std::move(buf1)};
        CHECK(read(buf2) == 1);
    } // both buffers destruct fine here

    // move assignment (via swap)
    {
        auto buf1 = buftest::allocBuf<DimExtent, Elem, Idx>(dev, extent);
        auto buf2 = buftest::allocBuf<DimExtent, Elem, Idx>(dev, extent);
        write(buf1, 1);
        write(buf2, 2);
        using std::swap;
        swap(buf1, buf2);
        CHECK(read(buf1) == 2);
        CHECK(read(buf2) == 1);
    } // both buffers destruct fine here
}
#endif

//! Copy between views that use different index types, like a std::vector, whose index type is always
//! std::size_t, and a buffer allocated with the index type of the accelerator.
TEMPLATE_LIST_TEST_CASE("memBufMixedIdxCopyTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;

    // a std::vector is a one-dimensional view
    if constexpr(Dim::value == 1)
    {
        using Idx = aV2P::Idx<Acc>;
        using Elem = std::uint32_t;

        // an index type which is guaranteed to differ from the one used by the accelerator
        using OtherIdx = std::conditional_t<sizeof(Idx) == sizeof(std::uint32_t), std::uint64_t, std::uint32_t>;
        STATIC_REQUIRE(not std::is_same_v<Idx, OtherIdx>);
        STATIC_REQUIRE(std::is_same_v<aV2P::Idx<std::vector<Elem>>, std::size_t>);

        auto const platformHost = aV2P::PlatformCpu{};
        auto const devHost = aV2P::getDevByIdx(platformHost, 0);
        auto const platformAcc = aV2P::Platform<Acc>{};
        auto const dev = aV2P::getDevByIdx(platformAcc, 0);
        auto queue = aV2P::Queue<Acc, aV2P::Blocking>{dev};

        constexpr std::size_t size = 16u;
        std::vector<Elem> src(size);
        std::iota(std::begin(src), std::end(src), Elem{1});
        std::vector<Elem> dst(size);

        auto buf = aV2P::allocBuf<Elem, Idx>(dev, aV2P::Vec<Dim, Idx>{static_cast<Idx>(size)});

        // copy without an explicit extent, which is taken from the source view
        std::fill(std::begin(dst), std::end(dst), Elem{0});
        aV2P::memcpy(queue, buf, src);
        aV2P::memcpy(queue, dst, buf);
        aV2P::wait(queue);
        CHECK(dst == src);

        // copy with an explicit extent, expressed in either index type
        std::fill(std::begin(dst), std::end(dst), Elem{0});
        aV2P::memcpy(queue, buf, src, static_cast<Idx>(size));
        aV2P::memcpy(queue, dst, buf, size);
        aV2P::wait(queue);
        CHECK(dst == src);

        // copy to and from views using a third index type
        std::fill(std::begin(dst), std::end(dst), Elem{0});
        auto const srcView = aV2P::createView(devHost, src.data(), static_cast<OtherIdx>(size));
        auto dstView = aV2P::createView(devHost, dst.data(), static_cast<OtherIdx>(size));
        STATIC_REQUIRE(std::is_same_v<aV2P::Idx<decltype(dstView)>, OtherIdx>);
        aV2P::memcpy(queue, buf, srcView);
        aV2P::memcpy(queue, dstView, buf);
        aV2P::wait(queue);
        CHECK(dst == src);
    }
}

namespace
{

    struct DummyType
    {
        auto check() -> bool
        {
            return true;
        }
    };

} // namespace

TEMPLATE_LIST_TEST_CASE("memBufAccessors", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Idx = aV2P::Idx<Acc>;
    using Elem = std::size_t;
    using Dim = aV2P::Dim<Acc>;

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    // aV2P 0D limitation: alpaka 3.0.0 has no zero-dimensional memory support
    // (onHost::alloc rejects Vec<T,0> extents via dim()/x() in api/util.hpp:106;
    // View requires T_Extents::UniVec, mem/View.hpp:104).
    // Restore when alpaka 3 supports 0D Buf/View/alloc; re-enable together with the adapter.
#if 0
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    // accessors for scalar buffers
    {
        auto buf = buftest::allocBuf<aV2P::DimInt<0u>, Elem, Idx>(devHost, aV2P::Vec<aV2P::DimInt<0u>, Idx>{});
        *buf = 42u;
        CHECK(*buf == 42);

        auto buf2
            = buftest::allocBuf<aV2P::DimInt<0u>, DummyType, Idx>(devHost, aV2P::Vec<aV2P::DimInt<0u>, Idx>{});
        CHECK(buf2->check());
    }
#endif

    if constexpr(Dim::value == 1)
    {
        auto queue = aV2P::Queue<Acc, aV2P::Blocking>{dev};
        auto const extent = aV2P::Vec<Dim, Idx>{};
        auto foo = [](std::span<Elem>) { return true; };

        auto buf = buftest::allocBuf<Dim, Elem, Idx>(dev, extent);

        CHECK(buf.size() == extent[0]);
        CHECK(buf.begin() == buf.data());
        CHECK(buf.cbegin() == buf.data());
        CHECK(buf.end() == buf.data() + buf.size());
        CHECK(buf.cend() == buf.data() + buf.size());
        CHECK((buf.end() - buf.begin()) == static_cast<long>(buf.size()));
        CHECK((buf.cend() - buf.cbegin()) == static_cast<long>(buf.size()));
        CHECK(foo(buf));
    }
}

TEMPLATE_LIST_TEST_CASE("memBufAllocDeducedIdx", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Idx = aV2P::Idx<Acc>;
    using Elem = std::size_t;
    using Dim = aV2P::Dim<Acc>;
    auto const extent = aV2P::Vec<Dim, Idx>{};
    auto const devHost = aV2P::getDevByIdx(aV2P::PlatformCpu{}, 0);
    auto const devAcc = aV2P::getDevByIdx(aV2P::Platform<Acc>{}, 0);
    auto queue = aV2P::Queue<Acc, aV2P::Blocking>{devAcc};

    auto host_buf = aV2P::allocBuf<Elem>(devHost, extent);
    auto dev_buf = aV2P::allocBuf<Elem>(devAcc, extent);
    auto dev_buf_async = aV2P::allocAsyncBuf<Elem>(queue, extent);
    auto dev_buf_async_supported = aV2P::allocAsyncBufIfSupported<Elem>(queue, extent);
    auto buf_mapped = aV2P::allocMappedBuf<Elem>(devHost, aV2P::PlatformCpu{}, extent);
    auto buf_managed = aV2P::allocManagedBuf<Elem>(devHost, aV2P::PlatformCpu{}, extent);
}
