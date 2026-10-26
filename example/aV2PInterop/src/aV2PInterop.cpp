/* Copyright 2026 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

// The point of this example is that both API headers are used in one
// translation unit and both operate on the same allocation.
#include <alpaka/alpaka.hpp>

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExampleDefaultAcc.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
    using Dim1 = aV2P::DimInt<1u>;
    using Dim2 = aV2P::DimInt<2u>;
    using Idx = std::size_t;
    using Data = std::uint32_t;

    using Acc = aV2P::ExampleDefaultAcc<Dim1, Idx>;
    using DevAcc = aV2P::Dev<Acc>;
    using QueueAcc = aV2P::Queue<Acc, aV2P::Blocking>;
    using EventAcc = aV2P::Event<QueueAcc>;

    bool g_ok = true;

    void report(bool condition, std::string const& what)
    {
        std::cout << (condition ? "PASS: " : "FAIL: ") << what << '\n';
        g_ok = g_ok && condition;
    }

    //! Native alpaka 3 in-place transform, stage one.
    struct AddValueKernel
    {
        template<typename TAcc, alpaka::concepts::IMdSpan TData, typename TExtent>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, TData data, TExtent const& extent, std::uint32_t value) const
        {
            for(auto idx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{extent}))
                data[idx] = data[idx] + value;
        }
    };

    //! Native alpaka 3 in-place transform, stage two.
    //!
    //! It reads the value written by stage one, so it only produces the expected
    //! result when both kernels are executed in enqueue order.
    struct MultiplyValueKernel
    {
        template<typename TAcc, alpaka::concepts::IMdSpan TData, typename TExtent>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, TData data, TExtent const& extent, std::uint32_t value) const
        {
            for(auto idx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{extent}))
                data[idx] = data[idx] * value;
        }
    };

    template<typename TBuf>
    auto extentsMatch(TBuf const& buf) -> bool
    {
        auto const aV2PExtents = aV2P::getExtents(buf);
        auto const nativeExtents = buf.getAlpaka3().getExtents();
        bool same = true;
        for(std::size_t i = 0u; i < aV2P::Dim<TBuf>::value; ++i)
            same = same && static_cast<std::size_t>(aV2PExtents[i]) == static_cast<std::size_t>(nativeExtents[i]);
        return same;
    }

    template<typename TBuf>
    auto pitchesMatch(TBuf const& buf) -> bool
    {
        auto const aV2PPitches = aV2P::getPitchesInBytes(buf);
        auto const nativePitches = buf.getAlpaka3().getPitches();
        bool same = true;
        for(std::size_t i = 0u; i < aV2P::Dim<TBuf>::value; ++i)
            same = same && static_cast<std::size_t>(aV2PPitches[i]) == static_cast<std::size_t>(nativePitches[i]);
        return same;
    }

    template<typename TBuf>
    void checkIdentity(TBuf const& buf, std::string const& name, Data const* baseBefore)
    {
        auto const* aV2PPtr = aV2P::getPtrNative(buf);
        auto const* nativePtr = buf.getAlpaka3().data();
        auto const* nativeViewPtr = alpaka::onHost::data(buf.getAlpaka3());

        report(aV2PPtr == nativePtr, name + ": aV2P data() equals the native buffer base pointer");
        report(aV2PPtr == nativeViewPtr, name + ": aV2P data() equals the native view base pointer");
        report(baseBefore == nullptr || baseBefore == nativePtr, name + ": buffer allocation was not recreated");
        report(extentsMatch(buf), name + ": aV2P and native extents match");
        report(pitchesMatch(buf), name + ": aV2P and native pitches match");
    }

    void testOneDimensional(DevAcc& dev, QueueAcc& queue, EventAcc& fillEvent)
    {
        constexpr Idx numElements = 16u;
        constexpr std::uint32_t addend = 5u;
        constexpr std::uint32_t factor = 3u;

        auto const devHost = aV2P::getDevByIdx(aV2P::PlatformCpu{}, 0);
        auto const extent = aV2P::Vec<Dim1, Idx>{numElements};

        using BufAcc = aV2P::Buf<DevAcc, Data, Dim1, Idx>;
        using BufHost = aV2P::Buf<aV2P::DevCpu, Data, Dim1, Idx>;

        BufHost hostSrc(aV2P::allocBuf<Data, Idx>(devHost, extent));
        BufHost hostResult(aV2P::allocBuf<Data, Idx>(devHost, extent));
        BufAcc buf(aV2P::allocBuf<Data, Idx>(dev, extent));

        for(Idx i = 0u; i < numElements; ++i)
            hostSrc[i] = static_cast<Data>(i + 1u);

        auto& nativeBuf = buf.getAlpaka3();
        Data const* baseBefore = nativeBuf.data();

        // aV2P writes the initial content.
        aV2P::memcpy(queue, buf, hostSrc);
        aV2P::enqueue(queue, fillEvent);
        aV2P::wait(fillEvent);
        checkIdentity(buf, "1D", baseBefore);

        // Native alpaka 3 queue on the device returned by the accessor.
        auto nativeQueue = dev.getAlpaka3().makeQueue();
        report(nativeQueue.getDevice() == dev.getAlpaka3(), "1D: native queue belongs to the aV2P device");

        auto const frameSpec
            = alpaka::onHost::FrameSpec{alpaka::Vec{std::uint32_t{4u}}, alpaka::Vec{std::uint32_t{4u}}};

        // Both stages use the same native queue. Nothing is copied or
        // synchronized on the aV2P side between them.
        nativeQueue.enqueue(frameSpec, AddValueKernel{}, nativeBuf, nativeBuf.getExtents(), addend);
        nativeQueue.enqueue(frameSpec, MultiplyValueKernel{}, nativeBuf, nativeBuf.getExtents(), factor);
        alpaka::onHost::wait(nativeQueue);

        auto expected = [&](Idx i) -> Data { return static_cast<Data>((hostSrc[i] + addend) * factor); };

        // Observe the ordered result through a native host copy. This does not
        // touch the aV2P queue or use an aV2P copy.
        auto nativeHost = alpaka::onHost::allocHost<Data>(nativeBuf.getExtents());
        alpaka::onHost::memcpy(nativeQueue, nativeHost, nativeBuf);
        alpaka::onHost::wait(nativeQueue);
        bool ordered = true;
        for(Idx i = 0u; i < numElements; ++i)
            ordered = ordered && nativeHost[i] == expected(i);
        report(ordered, "1D: native stage 2 observed stage 1 output (native queue is ordered)");

        // Read the same data back through aV2P after synchronizing the native queue.
        aV2P::memcpy(queue, hostResult, buf);
        aV2P::wait(queue);
        bool aV2PRead = true;
        for(Idx i = 0u; i < numElements; ++i)
            aV2PRead = aV2PRead && hostResult[i] == expected(i);
        report(aV2PRead, "1D: aV2P reads the native kernel result with exact values");
        checkIdentity(buf, "1D after native work", baseBefore);

        // Reverse direction: write through aV2P and read through native alpaka 3.
        aV2P::fill(queue, buf, Data{9u});
        aV2P::enqueue(queue, fillEvent);
        aV2P::wait(fillEvent);

        alpaka::onHost::memcpy(nativeQueue, nativeHost, nativeBuf);
        alpaka::onHost::wait(nativeQueue);
        bool reverse = true;
        for(Idx i = 0u; i < numElements; ++i)
            reverse = reverse && nativeHost[i] == Data{9u};
        report(reverse, "1D: native view observes the aV2P fill");
    }

    void testTwoDimensional(DevAcc& dev, QueueAcc& queue, EventAcc& fillEvent)
    {
        constexpr Idx nx = 4u;
        constexpr Idx ny = 3u;
        constexpr std::uint32_t addend = 7u;
        constexpr std::uint32_t factor = 2u;

        auto const devHost = aV2P::getDevByIdx(aV2P::PlatformCpu{}, 0);
        auto const extent = aV2P::Vec<Dim2, Idx>{nx, ny};

        using BufAcc = aV2P::Buf<DevAcc, Data, Dim2, Idx>;
        using BufHost = aV2P::Buf<aV2P::DevCpu, Data, Dim2, Idx>;

        BufHost hostSrc(aV2P::allocBuf<Data, Idx>(devHost, extent));
        BufHost hostResult(aV2P::allocBuf<Data, Idx>(devHost, extent));
        BufAcc buf(aV2P::allocBuf<Data, Idx>(dev, extent));

        for(Idx y = 0u; y < ny; ++y)
            for(Idx x = 0u; x < nx; ++x)
                hostSrc[aV2P::Vec<Dim2, Idx>{x, y}] = static_cast<Data>(10u + x + 5u * y);

        auto& nativeBuf = buf.getAlpaka3();
        Data const* baseBefore = nativeBuf.data();

        aV2P::memcpy(queue, buf, hostSrc);
        aV2P::enqueue(queue, fillEvent);
        aV2P::wait(fillEvent);
        checkIdentity(buf, "2D", baseBefore);

        auto nativeQueue = dev.getAlpaka3().makeQueue();
        report(nativeQueue.getDevice() == dev.getAlpaka3(), "2D: native queue belongs to the aV2P device");

        auto const frameSpec = alpaka::onHost::FrameSpec{
            alpaka::Vec{std::uint32_t{2u}, std::uint32_t{2u}},
            alpaka::Vec{std::uint32_t{2u}, std::uint32_t{2u}}};

        nativeQueue.enqueue(frameSpec, AddValueKernel{}, nativeBuf, nativeBuf.getExtents(), addend);
        nativeQueue.enqueue(frameSpec, MultiplyValueKernel{}, nativeBuf, nativeBuf.getExtents(), factor);
        alpaka::onHost::wait(nativeQueue);

        auto expected = [&](Idx x, Idx y) -> Data
        { return static_cast<Data>((hostSrc[aV2P::Vec<Dim2, Idx>{x, y}] + addend) * factor); };

        auto nativeHost = alpaka::onHost::allocHost<Data>(nativeBuf.getExtents());
        alpaka::onHost::memcpy(nativeQueue, nativeHost, nativeBuf);
        alpaka::onHost::wait(nativeQueue);
        bool ordered = true;
        for(Idx y = 0u; y < ny; ++y)
            for(Idx x = 0u; x < nx; ++x)
                ordered = ordered && nativeHost[alpaka::Vec<Idx, 2u>{x, y}] == expected(x, y);
        report(ordered, "2D: native stage 2 observed stage 1 output (native queue is ordered)");

        aV2P::memcpy(queue, hostResult, buf);
        aV2P::wait(queue);
        bool aV2PRead = true;
        for(Idx y = 0u; y < ny; ++y)
            for(Idx x = 0u; x < nx; ++x)
                aV2PRead = aV2PRead && hostResult[aV2P::Vec<Dim2, Idx>{x, y}] == expected(x, y);
        report(aV2PRead, "2D: aV2P reads the native kernel result with exact values");
        checkIdentity(buf, "2D after native work", baseBefore);

        aV2P::fill(queue, buf, Data{11u});
        aV2P::enqueue(queue, fillEvent);
        aV2P::wait(fillEvent);

        alpaka::onHost::memcpy(nativeQueue, nativeHost, nativeBuf);
        alpaka::onHost::wait(nativeQueue);
        bool reverse = true;
        for(Idx y = 0u; y < ny; ++y)
            for(Idx x = 0u; x < nx; ++x)
                reverse = reverse && nativeHost[alpaka::Vec<Idx, 2u>{x, y}] == Data{11u};
        report(reverse, "2D: native view observes the aV2P fill");
    }
} // namespace

auto main() -> int
{
    auto dev = aV2P::getDevByIdx(aV2P::Platform<Acc>{}, 0);
    QueueAcc queue(dev);
    EventAcc fillEvent(dev);

    std::cout << "Using aV2P accelerator: " << aV2P::getAccName<Acc>() << '\n';

    // Identity of the wrappers themselves: the accessor returns references into
    // the existing queue and event and never creates replacements.
    auto* queueAddress = &queue.getAlpaka3();
    auto* eventAddress = &fillEvent.getAlpaka3();

    testOneDimensional(dev, queue, fillEvent);
    testTwoDimensional(dev, queue, fillEvent);

    report(&queue.getAlpaka3() == queueAddress, "aV2P queue accessor did not recreate the queue");
    report(&fillEvent.getAlpaka3() == eventAddress, "aV2P event accessor did not recreate the event");
    report(queue.getAlpaka3().getDevice() == dev.getAlpaka3(), "aV2P queue still belongs to the aV2P device");

    if(g_ok)
    {
        std::cout << "PASS: aV2P and native alpaka 3 shared one resource\n";
        return EXIT_SUCCESS;
    }

    std::cout << "FAIL: aV2P and native alpaka 3 interoperability check failed\n";
    return EXIT_FAILURE;
}
