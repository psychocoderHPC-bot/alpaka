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

struct LambdaKernelWorkDiv
{
    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& acc, bool* success) const
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<TAcc>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());
    }
};

struct LambdaKernelWithArgument
{
    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& /* acc */, bool* success, std::uint32_t const& arg1) const
    {
        AV2P_CHECK(*success, 42u == arg1);
    }
};

struct LambdaKernelWithCapture
{
    std::uint32_t arg;

    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& /* acc */, bool* success) const
    {
        AV2P_CHECK(*success, 42u == arg);
    }
};

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
        LambdaKernelWorkDiv kernel;
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
        LambdaKernelWithArgument kernel;

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

        LambdaKernelWithCapture kernel{arg};

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
