/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

//! Regression tests for the 2.2 contiguous buffer layout contract on top of alpaka 3 allocations.
//!
//! alpaka 3 host allocations pad the innermost row pitch to the backend SIMD alignment.  aV2P
//! re-views the same allocation with contiguous 2.2 pitches.  These tests pin the observable
//! layout (pitches, linear `std::data` access), the interoperability of `getAlpaka3()` with native
//! alpaka 3 operations in both directions, and that aV2P operations never touch the padding bytes
//! beyond the contiguous span.

#include <aV2P/alpaka.hpp>
#include <aV2P/mem/buf/Traits.hpp>
#include <aV2P/mem/view/Traits.hpp>

#include <catch2/catch_test_macros.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/api/util.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace
{
    using Idx = std::uint32_t;
    using Elem = std::uint32_t;

    auto makeHostCpuDev()
    {
        return aV2P::getDevByIdx(aV2P::PlatformCpu{}, 0);
    }

    //! Compare the aV2P pitches with the contiguous 2.2 pitches computed from the extents.
    template<typename TBuf>
    void checkContiguousPitches(TBuf const& buf)
    {
        auto const extents = aV2P::getExtents(buf);
        auto const expected = aV2P::detail::calculatePitchesFromExtents<Elem>(extents);
        auto const actual = aV2P::getPitchesInBytes(buf);
        for(std::size_t i = 0; i < aV2P::Dim<TBuf>::value; ++i)
            CHECK(actual[i] == expected[i]);
    }
} // namespace

TEST_CASE("memBufContiguousLayout2d", "[memBuf][layout]")
{
    auto const dev = makeHostCpuDev();
    auto const extent = aV2P::Vec<aV2P::DimInt<2>, Idx>{3u, 5u};
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);

    checkContiguousPitches(buf);

    // The 2.2 contract: pitches == extents * sizeof(T), i.e. 5*4 and 4 bytes.
    auto const pitches = aV2P::getPitchesInBytes(buf);
    CHECK(pitches[0] == 5u * sizeof(Elem));
    CHECK(pitches[1] == sizeof(Elem));

    // `std::data(buf)[i]` must span the whole buffer linearly and agree with `buf[Vec{y, x}]`.
    for(Idx y = 0; y < extent[0]; ++y)
        for(Idx x = 0; x < extent[1]; ++x)
            buf[aV2P::Vec{y, x}] = y * extent[1] + x;

    for(Idx i = 0; i < extent[0] * extent[1]; ++i)
        CHECK(std::data(buf)[i] == buf[aV2P::Vec{i / extent[1], i % extent[1]}]);
}

TEST_CASE("memBufContiguousLayout3d", "[memBuf][layout]")
{
    auto const dev = makeHostCpuDev();
    auto const extent = aV2P::Vec<aV2P::DimInt<3>, Idx>{2u, 3u, 5u};
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);

    checkContiguousPitches(buf);

    auto const pitches = aV2P::getPitchesInBytes(buf);
    CHECK(pitches[0] == 3u * 5u * sizeof(Elem));
    CHECK(pitches[1] == 5u * sizeof(Elem));
    CHECK(pitches[2] == sizeof(Elem));

    for(Idx z = 0; z < extent[0]; ++z)
        for(Idx y = 0; y < extent[1]; ++y)
            for(Idx x = 0; x < extent[2]; ++x)
                buf[aV2P::Vec{z, y, x}] = (z * extent[1] + y) * extent[2] + x;

    for(Idx i = 0; i < extent[0] * extent[1] * extent[2]; ++i)
        CHECK(
            std::data(buf)[i]
            == buf[aV2P::Vec{i / (extent[1] * extent[2]), (i / extent[2]) % extent[1], i % extent[2]}]);
}

TEST_CASE("memBufAlpaka3Interop", "[memBuf][layout]")
{
    auto const dev = makeHostCpuDev();
    auto const extent = aV2P::Vec<aV2P::DimInt<2>, Idx>{3u, 5u};
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);
    aV2P::QueueCpuBlocking queue{dev};

    // The handle a native alpaka 3 consumer sees must use the same contiguous pitches.
    auto const nativePitches = buf.getAlpaka3().getPitches();
    CHECK(nativePitches[0] == 5u * sizeof(Elem));
    CHECK(nativePitches[1] == sizeof(Elem));

    // (b) Native alpaka 3 element access on getAlpaka3() writes the elements aV2P reads.
    constexpr Elem nativeValue = 0x11223344u;
    auto& nativeView = buf.getAlpaka3();
    for(Idx y = 0; y < extent[0]; ++y)
        for(Idx x = 0; x < extent[1]; ++x)
            nativeView[alpaka::Vec<Idx, 2u>{y, x}] = nativeValue;
    for(Idx y = 0; y < extent[0]; ++y)
        for(Idx x = 0; x < extent[1]; ++x)
            CHECK(buf[aV2P::Vec{y, x}] == nativeValue);

    // (c) An aV2P fill is visible through the native alpaka 3 handle.
    constexpr Elem av2pValue = 0x55667788u;
    aV2P::fill(queue, buf, av2pValue);
    for(Idx y = 0; y < extent[0]; ++y)
        for(Idx x = 0; x < extent[1]; ++x)
            CHECK(buf.getAlpaka3()[alpaka::Vec<Idx, 2u>{y, x}] == av2pValue);
}

TEST_CASE("memBufNativeBulkFillUsesContiguousAlignment", "[memBuf][layout]")
{
    auto const dev = makeHostCpuDev();
    auto const extent = aV2P::Vec<aV2P::DimInt<2>, Idx>{4u, 5u};
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);
    aV2P::QueueCpuBlocking queue{dev};

    constexpr auto allocationAlignment = alpaka::api::util::simdOptimizedAlignment<Elem>(
        alpaka::api::Host{},
        alpaka::deviceKind::Cpu{});
    auto const rowBytes = static_cast<std::size_t>(extent[1]) * sizeof(Elem);

    // The regression is only observable when the contiguous row pitch is not a multiple of the
    // allocation alignment that the padded alpaka 3 layout used to guarantee for every row.
    REQUIRE(allocationAlignment > alignof(Elem));
    REQUIRE(rowBytes % allocationAlignment != 0u);

    // A contiguous multi-dimensional re-view must not advertise the padded allocation alignment.
    //
    // Query the alignment through the native buffer type instead of through the handle value.
    // `Alignment::get<Elem>()` is consteval, so its object expression is evaluated in a constant
    // expression; passing the non-constexpr `buf.getAlpaka3()` makes that expression non-constant.
    // clang rejects it and nvcc 12.5 ICEs in `diagnose_consteval_routine_node`.  The buffer's
    // alignment is a type-level property (a static constexpr member), so reading it from the type
    // is equivalent and keeps the assertion unchanged.
    using NativeBuffer = std::remove_cvref_t<decltype(buf.getAlpaka3())>;
    CHECK(NativeBuffer::getAlignment().template get<Elem>() == alignof(Elem));

    // A native alpaka 3 SIMD fill must not assume aligned interior rows of this re-view.
    constexpr Elem fillValue = 0xA1B2C3D4u;
    alpaka::onHost::fill(queue.getAlpaka3(), buf.getAlpaka3(), fillValue);
    aV2P::wait(queue);

    for(Idx y = 0; y < extent[0]; ++y)
        for(Idx x = 0; x < extent[1]; ++x)
        {
            auto const nativeIdx = alpaka::Vec<Idx, 2u>{y, x};
            CHECK(buf.getAlpaka3()[nativeIdx] == fillValue);
            CHECK(buf[aV2P::Vec{y, x}] == fillValue);
        }

    // The native byte-wise memset uses the contiguous pitches as well.
    constexpr std::uint8_t byteValue = 0x5Au;
    alpaka::onHost::memset(queue.getAlpaka3(), buf.getAlpaka3(), byteValue);
    aV2P::wait(queue);

    Elem expected{};
    std::memset(&expected, byteValue, sizeof(Elem));
    for(Idx y = 0; y < extent[0]; ++y)
        for(Idx x = 0; x < extent[1]; ++x)
        {
            auto const nativeIdx = alpaka::Vec<Idx, 2u>{y, x};
            CHECK(buf.getAlpaka3()[nativeIdx] == expected);
            CHECK(buf[aV2P::Vec{y, x}] == expected);
        }
}

TEST_CASE("memBufDoesNotTouchPadding", "[memBuf][layout]")
{
    auto const dev = makeHostCpuDev();
    auto const extent = aV2P::Vec<aV2P::DimInt<2>, Idx>{3u, 5u};
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);
    aV2P::QueueCpuBlocking queue{dev};

    // Reconstruct the size and pitches of the padded alpaka 3 allocation that backs the re-view.
    constexpr auto alignment = alpaka::api::util::simdOptimizedAlignment<Elem>(
        alpaka::api::Host{},
        alpaka::deviceKind::Cpu{});
    auto const [paddedBytes, paddedPitches] = alpaka::api::util::emulatedAlignedMemDescription<Elem>(
        alignment,
        buf.getAlpaka3().getExtents());

    auto const contiguousBytes = static_cast<std::size_t>(extent[0]) * extent[1] * sizeof(Elem);
    REQUIRE(contiguousBytes <= paddedBytes);
    // This configuration must actually be padded, otherwise the guard has nothing to observe.
    REQUIRE(paddedPitches[0] > extent[1] * sizeof(Elem));

    // Poison the padding/tail bytes beyond the contiguous span, inside the allocation.
    constexpr std::uint8_t poison = 0x5Au;
    auto* raw = reinterpret_cast<std::uint8_t*>(std::data(buf));
    for(std::size_t i = contiguousBytes; i < paddedBytes; ++i)
        raw[i] = poison;

    // aV2P operations must write the contiguous span only.
    aV2P::fill(queue, buf, Elem{0x01020304u});
    aV2P::memset(queue, buf, std::uint8_t{0xFFu});
    aV2P::wait(queue);

    for(std::size_t i = 0; i < contiguousBytes; ++i)
        CHECK(raw[i] == 0xFFu);
    for(std::size_t i = contiguousBytes; i < paddedBytes; ++i)
        CHECK(raw[i] == poison);
}
