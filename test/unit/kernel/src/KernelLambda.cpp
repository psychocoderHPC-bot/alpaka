/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

// NVCC needs --expt-extended-lambda
#if !defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__))

#    include <aV2P/core/Config.hpp>
#    include <aV2P/kernel/Traits.hpp>
#    include <aV2P/test/KernelExecutionFixture.hpp>
#    include <aV2P/test/acc/TestAccs.hpp>

#    include <catch2/catch_test_macros.hpp>

struct TestTemplateLambda
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;

        aV2P::test::KernelExecutionFixture<TAcc> fixture(aV2P::Vec<Dim, Idx>::ones());

#    if AV2P_COMP_MSVC
#        pragma warning(push)
#        pragma warning(disable : 4702) // warning C4702: unreachable code
#    endif
        auto kernel = [] AV2P_FN_ACC(TAcc const& acc, bool* success) -> void
        {
            AV2P_CHECK(
                *success,
                static_cast<aV2P::Idx<TAcc>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());
        };
#    if AV2P_COMP_MSVC
#        pragma warning(pop)
#    endif

        REQUIRE(fixture(kernel));
    }
};

struct TestTemplateArg
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;

        aV2P::test::KernelExecutionFixture<TAcc> fixture(aV2P::Vec<Dim, Idx>::ones());

        std::uint32_t const arg = 42u;
        auto kernel = [] AV2P_FN_ACC(TAcc const& /* acc */, bool* success, std::uint32_t const& arg1) -> void
        { AV2P_CHECK(*success, 42u == arg1); };

        REQUIRE(fixture(kernel, arg));
    }
};

struct TestTemplateCapture
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;

        aV2P::test::KernelExecutionFixture<TAcc> fixture(aV2P::Vec<Dim, Idx>::ones());

        std::uint32_t arg = 42u;

        auto kernel = [arg] AV2P_FN_ACC(TAcc const& /* acc */, bool* success) -> void
        { AV2P_CHECK(*success, 42u == arg); };

        REQUIRE(fixture(kernel));
    }
};

TEST_CASE("lambdaKernelIsWorking", "[kernel]")
{
    aV2P::meta::forEachType<aV2P::test::TestAccs>(TestTemplateLambda());
}

TEST_CASE("lambdaKernelWithArgumentIsWorking", "[kernel]")
{
    aV2P::meta::forEachType<aV2P::test::TestAccs>(TestTemplateArg());
}

TEST_CASE("lambdaKernelWithCapturingIsWorking", "[kernel]")
{
    aV2P::meta::forEachType<aV2P::test::TestAccs>(TestTemplateCapture());
}

#endif
