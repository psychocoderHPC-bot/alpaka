/* Copyright 2023 Jeffrey Kelling, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>
#include <aV2P/rand/RandPhiloxStateless.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>

#ifdef AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED

//! A kernel that fills an array with pseudo-random numbers.
class CounterBasedRngKernel
{
public:
    template<class TAcc>
    using Vec = aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>>;
    using Gen = typename aV2P::rand::PhiloxStateless4x32x10Vector;
    using Key = typename Gen::Key;
    using Counter = typename Gen::Counter;

    template<typename TAcc, typename TElem>
    using Mdspan = aV2P::experimental::MdSpan<TElem, aV2P::Idx<TAcc>, aV2P::Dim<TAcc>>;

private:
    template<unsigned int I>
    struct ElemLoop
    {
        AV2P_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TElem>
        static AV2P_FN_ACC auto elemLoop(
            TAcc const& acc,
            Mdspan<TAcc, TElem> dst,
            Key const& key,
            Vec<TAcc> const& threadElemExtent,
            Vec<TAcc>& threadFirstElemIdx) -> void
        {
            auto const threadLastElemIdx = threadFirstElemIdx[I] + threadElemExtent[I];
            auto const threadLastElemIdxClipped
                = (dst.extent(I) > threadLastElemIdx) ? threadLastElemIdx : dst.extent(I);

            constexpr auto Dim = aV2P::Dim<TAcc>::value;

            auto const firstElem = threadFirstElemIdx[I];
            if constexpr(I < Dim - 1)
            {
                for(; threadFirstElemIdx[I] < threadLastElemIdxClipped; ++threadFirstElemIdx[I])
                {
                    ElemLoop<I + 1>::elemLoop(acc, dst, key, threadElemExtent, threadFirstElemIdx);
                }
            }
            else
            {
                Counter c = {0, 0, 0, 0};
                for(unsigned int i = 0; i < Dim; ++i)
                    c[i] = threadFirstElemIdx[i];

                for(; threadFirstElemIdx[Dim - 1] < threadLastElemIdxClipped; ++threadFirstElemIdx[Dim - 1])
                {
                    c[Dim - 1] = threadFirstElemIdx[Dim - 1];
                    auto const random = Gen::generate(c, key);
                    // to make use of the whole random vector we would need to ensure numElement[0] % 4 == 0
                    dst(aV2P::toArray(threadFirstElemIdx)) = TElem(random[0]);
                }
            }
            threadFirstElemIdx[I] = firstElem;
        }
    };

public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param dst destimation matrix.
    //! \param extent The matrix dimension in elements.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem>
    AV2P_FN_ACC auto operator()(TAcc const& acc, Mdspan<TAcc, TElem> dst, Key const& key) const -> void
    {
        constexpr auto Dim = aV2P::Dim<TAcc>::value;
        static_assert(Dim <= 4, "The CounterBasedRngKernel expects at most 4-dimensional indices!");

        Vec<TAcc> const gridThreadIdx(aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc));
        Vec<TAcc> const threadElemExtent(aV2P::getWorkDiv<aV2P::Thread, aV2P::Elems>(acc));
        Vec<TAcc> threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        ElemLoop<0>::elemLoop(acc, dst, key, threadElemExtent, threadFirstElemIdx);
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the index domain
    using Dim = aV2P::DimInt<3u>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = aV2P::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << aV2P::getAccName<Acc>() << std::endl;

    using AccHost = aV2P::AccCpuSerial<Dim, Idx>;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = aV2P::Blocking;
    using QueueAcc = aV2P::Queue<Acc, QueueProperty>;
    using QueueHost = aV2P::Queue<AccHost, QueueProperty>;

    // Select a device
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    QueueAcc queueAcc(devAcc);
    QueueHost queueHost(devHost);

    // Define the work division
    aV2P::Vec<Dim, Idx> const extent = {16, 16, 16 * 8};
    aV2P::Vec<Dim, Idx> const elementsPerThread = {1, 1, 1};
    aV2P::Vec<Dim, Idx> const elementsPerThreadHost = {1, 1, 8};


    // Define the buffer element type
    using Data = std::uint32_t;

    // Allocate 3 host memory buffers
    auto bufHost(aV2P::allocBuf<Data, Idx>(devHost, extent));
    auto bufHostDev(aV2P::allocBuf<Data, Idx>(devHost, extent));

    std::random_device rd{};
    CounterBasedRngKernel::Key key = {rd(), rd()};

    // Allocate buffer on the accelerator
    using BufAcc = aV2P::Buf<Acc, Data, Dim, Idx>;
    BufAcc bufAcc(aV2P::allocBuf<Data, Idx>(devAcc, extent));

    CounterBasedRngKernel counterBasedRngKernel;

    // Let alpaka calculate good block and grid sizes given our full problem extent
    aV2P::KernelCfg<Acc> kernerlCfgAccDev = {extent, elementsPerThread};
    auto const workDivAcc = aV2P::getValidWorkDiv(
        kernerlCfgAccDev,
        devAcc,
        counterBasedRngKernel,
        aV2P::experimental::getMdSpan(bufAcc),
        key);

    // Create the kernel execution task.
    auto const taskKernelAcc = aV2P::createTaskKernel<Acc>(
        workDivAcc,
        CounterBasedRngKernel(),
        aV2P::experimental::getMdSpan(bufAcc),
        key);

    aV2P::KernelCfg<AccHost> kernerlCfgAccHost = {extent, elementsPerThreadHost};
    auto const workDivHost = aV2P::getValidWorkDiv(
        kernerlCfgAccHost,
        devHost,
        counterBasedRngKernel,
        aV2P::experimental::getMdSpan(bufHost),
        key);

    auto const taskKernelHost = aV2P::createTaskKernel<AccHost>(
        workDivHost,
        CounterBasedRngKernel(),
        aV2P::experimental::getMdSpan(bufHost),
        key);

    // Enqueue the kernel execution task
    aV2P::enqueue(queueHost, taskKernelHost);
    aV2P::enqueue(queueAcc, taskKernelAcc);

    // Copy the result from the device
    aV2P::memcpy(queueAcc, bufHostDev, bufAcc);
    auto const numElements = extent.prod();

    // wait in case we are using an asynchronous queue to time actual kernel runtime
    aV2P::wait(queueHost);
    aV2P::wait(queueAcc);

    int falseResults = 0;
    int const maxPrintFalseResults = extent[2] * 2;

    auto aHost = aV2P::experimental::getMdSpan(bufHost);
    auto aAcc = aV2P::experimental::getMdSpan(bufHostDev);
    for(Idx z = 0; z < aHost.extent(0); ++z)
        for(Idx y = 0; y < aHost.extent(1); ++y)
            for(Idx x = 0; x < aHost.extent(2); ++x)
            {
                Data const& valHost(aHost(z, y, x));
                Data const& valAcc(aAcc(z, y, x));
                if(valHost != valAcc)
                {
                    if(falseResults < maxPrintFalseResults)
                        std::cerr << "host[" << z << ", " << y << ", " << x << "] = " << valHost << " != acc[" << z
                                  << ", " << y << ", " << x << "] = " << valAcc << std::endl;
                    ++falseResults;
                }
            }

    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << maxPrintFalseResults
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    std::cout << "Check enabled accelerator tags:" << std::endl;
    aV2P::printTagNames<aV2P::EnabledAccTags>();
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagGenericSycl, TagGpuSyclIntel
    return aV2P::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}

#else

int main()
{
    return EXIT_SUCCESS;
}

#endif
