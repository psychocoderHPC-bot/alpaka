/* Copyright 2022  René Widera, Mehmet Yusufoglu, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/RuntimeMacros.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class KernelWithThrow
{
public:
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        using Idx = aV2P::Idx<TAcc>;
        using Dim = aV2P::Dim<TAcc>;
        using Vec = aV2P::Vec<Dim, Idx>;
        Vec const globalThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc);
        if(globalThreadIdx == Vec::zeros())
        {
            // Throw abort or std::runtime_error depending on acc type
            AV2P_THROW_ACC("Exception thrown by the kernel.");
        }
        aV2P::syncBlockThreads(acc);
    }
};

template<typename T, typename Acc>
void checkThrow(std::string const& expectedErrStr)
{
    if constexpr(aV2P::accMatchesTags<Acc, T>)
    {
        using Idx = aV2P::Idx<Acc>;
        using Dim = aV2P::Dim<Acc>;
        using Vec = aV2P::Vec<Dim, Idx>;
        using Queue = aV2P::Queue<Acc, aV2P::Blocking>;

        auto const platformAcc = aV2P::Platform<Acc>{};
        auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

        Queue queue(devAcc);
        auto const workDiv = aV2P::WorkDivMembers<Dim, Idx>{Vec{8}, Vec{1}, Vec{1}};

        try
        {
            aV2P::exec<Acc>(queue, workDiv, KernelWithThrow{});
            // Cuda can catch exceptions which were thrown at kernel during the wait(); therefore wait is added.
            aV2P::wait(queue);
        }
        catch(std::runtime_error& e)
        {
            std::string const errorStr{e.what()};
            printf("The error str catched: %s \n", errorStr.c_str());
            printf("The expected str in error str: %s \n", expectedErrStr.c_str());

            auto const found = errorStr.find(expectedErrStr);
            CHECK(found != std::string::npos);
        }
        catch(std::exception& e)
        {
            FAIL(std::string("Wrong exception type thrown in kernel:") + e.what());
        }
    }
}

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("ThrowForCpuThreadAndSerial", "[runtime]", TestAccs)
{
    using Acc = TestType;
    // Test runtime-error exceptions.
    checkThrow<aV2P::TagCpuThreads, Acc>("Exception thrown by the kernel");
    checkThrow<aV2P::TagCpuSerial, Acc>("Exception thrown by the kernel");
}

TEMPLATE_LIST_TEST_CASE("ThrowForGpuBackend", "[runtime]", TestAccs)
{
    using Acc = TestType;
    // Test runtime-error exceptions.
    checkThrow<aV2P::TagGpuCudaRt, Acc>("cudaErrorLaunchFailure");
}
