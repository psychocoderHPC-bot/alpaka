/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Jakob Krude, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/DemangleTypeNames.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/Iterator.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#if AV2P_COMP_MSVC
#    pragma warning(push)
#    pragma warning(disable : 4127) // suppress warning for c++17 conditional expression is constant
#endif

template<typename TDim, typename TIdx, typename TAcc, typename TData, typename Vec = aV2P::Vec<TDim, TIdx>>
struct TestContainer
{
    using AccQueueProperty = aV2P::Blocking;
    using DevQueue = aV2P::Queue<TAcc, AccQueueProperty>;
    using DevAcc = aV2P::Dev<TAcc>;
    using PlatformAcc = aV2P::Platform<TAcc>;

    using DevHost = aV2P::DevCpu;
    using PlatformHost = aV2P::Platform<DevHost>;

    using BufHost = aV2P::Buf<DevHost, TData, TDim, TIdx>;
    using BufDevice = aV2P::Buf<DevAcc, TData, TDim, TIdx>;

    using SubView = aV2P::ViewSubView<DevAcc, TData, TDim, TIdx>;

    PlatformAcc platformAcc{};
    DevAcc devAcc{aV2P::getDevByIdx(platformAcc, 0)};
    PlatformHost platformHost{};
    DevHost devHost{aV2P::getDevByIdx(platformHost, 0u)};
    DevQueue devQueue{devAcc};

    auto createHostBuffer(Vec extents, bool indexed) -> BufHost
    {
        BufHost bufHost(aV2P::allocBuf<TData, TIdx>(devHost, extents));
        if(indexed)
        {
            TData* const ptr = bufHost.data();
            for(TIdx i = 0; i < extents.prod(); ++i)
            {
                ptr[i] = static_cast<TData>(i);
            }
        }
        return bufHost;
    }

    auto createDeviceBuffer(Vec extents) -> BufDevice
    {
        BufDevice bufDevice(aV2P::allocBuf<TData, TIdx>(devAcc, extents));
        return bufDevice;
    }

    auto copyToAcc(BufHost bufHost, BufDevice bufAcc, Vec extents) -> void
    {
        aV2P::memcpy(devQueue, bufAcc, bufHost, extents);
    }

    auto copyToHost(BufDevice bufAcc, BufHost bufHost, Vec extents) -> void
    {
        aV2P::memcpy(devQueue, bufHost, bufAcc, extents);
    }

    auto copySliceOnDevice(BufDevice bufferToBeSliced, Vec subViewExtents, Vec offsets) -> BufDevice
    {
        BufDevice slicedBuffer = createDeviceBuffer(subViewExtents);
        // Create a subView with a possible offset.
        SubView subView = SubView(bufferToBeSliced, subViewExtents, offsets);
        // Copy the subView into a new buffer.
        aV2P::memcpy(devQueue, slicedBuffer, subView, subViewExtents);
        return slicedBuffer;
    }

    auto zeroSliceOnDevice(BufDevice bufferToBeSliced, Vec subViewExtents, Vec offsets) -> void
    {
        // Create a subView with a possible offset.
        SubView subView = SubView(bufferToBeSliced, subViewExtents, offsets);
        // Fill the subView with zeros.
        aV2P::memset(devQueue, subView, 0u);
    }

    auto compareBuffer(BufHost const& bufferA, BufHost const& bufferB, Vec const& extents) const
    {
        TData const* const ptrA = bufferA.data();
        TData const* const ptrB = bufferB.data();
        for(TIdx i = 0; i < extents.prod(); ++i)
        {
            INFO("Dim: " << TDim::value);
            INFO("Idx: " << aV2P::core::demangled<TIdx>);
            INFO("Acc: " << aV2P::trait::GetAccName<TAcc>::getAccName());
            INFO("i: " << i);
            REQUIRE(ptrA[i] == Catch::Approx(ptrB[i]));
        }
    }
};

using DataTypes = std::tuple<int, float, double>;

using TestAccWithDataTypes = aV2P::meta::CartesianProduct<std::tuple, aV2P::test::TestAccs, DataTypes>;

TEMPLATE_LIST_TEST_CASE("memBufSlicingMemcpyTest", "[memBuf]", TestAccWithDataTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using Data = std::tuple_element_t<1, TestType>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    // Test only buffer slices with up to three dimensions.
    if constexpr(Dim::value < 4)
    {
        TestContainer<Dim, Idx, Acc, Data> slicingTest;

        auto const extents = aV2P::test::extentBuf<Dim, Idx>;
        auto const extentsSubView = aV2P::test::extentSubView<Dim, Idx>;
        auto const offsets = aV2P::test::offset<Dim, Idx>;

        // This is the initial buffer.
        auto const indexedBuffer = slicingTest.createHostBuffer(extents, true);

        // This buffer will hold the copy of the initial buffer on the device.
        auto deviceBuffer = slicingTest.createDeviceBuffer(extents);

        // This buffer will hold the sliced-buffer when it is copied back to the host.
        auto resultBuffer = slicingTest.createHostBuffer(extentsSubView, false);

        // Copy the initial buffer to the device.
        slicingTest.copyToAcc(indexedBuffer, deviceBuffer, extents);

        // Make a copy of a slice of the buffer on the device.
        auto slicedBuffer = slicingTest.copySliceOnDevice(deviceBuffer, extentsSubView, offsets);

        // Copy the slice back to the host.
        slicingTest.copyToHost(slicedBuffer, resultBuffer, extentsSubView);

        // Compute the expected content of the slice.
        using Dim1 = aV2P::DimInt<1u>;
        auto correctResults = slicingTest.createHostBuffer(extentsSubView, false);
        Data* const ptr = correctResults.data();
        for(Idx i = 0; i < extentsSubView.prod(); ++i)
        {
            auto mappedToND = aV2P::mapIdx<Dim::value, Dim1::value>(aV2P::Vec<Dim1, Idx>(i), extentsSubView);
            auto addedOffset = mappedToND + offsets;
            auto mappedTo1D = aV2P::mapIdx<Dim1::value>(
                addedOffset,
                extents)[0]; // take the only element in the vector
            ptr[i] = static_cast<Data>(mappedTo1D);
        }

        // Compare the resultBuffer with the results computed manually.
        slicingTest.compareBuffer(resultBuffer, correctResults, extentsSubView);
    }
}

TEMPLATE_LIST_TEST_CASE("memBufSlicingMemsetTest", "[memBuf]", TestAccWithDataTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using Data = std::tuple_element_t<1, TestType>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    // Test only buffer slices with up to three dimensions.
    if constexpr(Dim::value < 4)
    {
        TestContainer<Dim, Idx, Acc, Data> slicingTest;

        auto const extents = aV2P::test::extentBuf<Dim, Idx>;
        auto const extentsSubView = aV2P::test::extentSubView<Dim, Idx>;
        auto const offsets = aV2P::test::offset<Dim, Idx>;

        // This is the initial buffer.
        auto const indexedBuffer = slicingTest.createHostBuffer(extents, true);

        // This buffer will hold the copy of the initial buffer on the device.
        auto deviceBuffer = slicingTest.createDeviceBuffer(extents);

        // This buffer will hold a copy of the initial buffer, after a slice has been set to zeroes.
        auto resultBuffer = slicingTest.createHostBuffer(extents, false);

        // Copy the initial buffer to the device.
        slicingTest.copyToAcc(indexedBuffer, deviceBuffer, extents);

        // Fill a slice of the buffer with zeroes.
        slicingTest.zeroSliceOnDevice(deviceBuffer, extentsSubView, offsets);

        // Copy the buffer back to the host
        slicingTest.copyToHost(deviceBuffer, resultBuffer, extents);

        // Compute the expected content of the buffer, with a slice set to zeroes.
        using Dim1 = aV2P::DimInt<1u>;
        auto correctResults = slicingTest.createHostBuffer(extents, true);
        Data* const ptr = correctResults.data();
        for(Idx i = 0; i < extents.prod(); ++i)
        {
            auto mappedToND = aV2P::mapIdx<Dim::value, Dim1::value>(aV2P::Vec<Dim1, Idx>(i), extents);
            if((mappedToND >= offsets && mappedToND < offsets + extentsSubView).all())
                ptr[i] = static_cast<Data>(0u);
        }

        // Compare the resultBuffer with the results computed manually.
        slicingTest.compareBuffer(resultBuffer, correctResults, extentsSubView);
    }
}

#if AV2P_COMP_MSVC
#    pragma warning(pop)
#endif
