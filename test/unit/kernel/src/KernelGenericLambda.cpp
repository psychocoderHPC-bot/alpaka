/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/kernel/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

// CUDA C Programming guide says: "__host__ __device__ extended lambdas cannot be generic lambdas"
#if !defined(AV2P_ACC_GPU_CUDA_ENABLED)

struct GenericLambdaKernel
{
    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& acc, bool* success) const
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<TAcc>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());
    }
};

template<typename TAcc>
struct VariadicGenericLambdaKernel
{
    template<typename... TArgs>
    AV2P_FN_ACC void operator()(TAcc const& /* acc */, bool* success, TArgs... args) const
    {
        AV2P_CHECK(*success, aV2P::meta::foldr([](auto a, auto b) { return a + b; }, args...) == (42u + 43u));
    }
};

TEMPLATE_LIST_TEST_CASE("genericLambdaKernelIsWorking", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    GenericLambdaKernel kernel;

    REQUIRE(fixture(kernel));
}

TEMPLATE_LIST_TEST_CASE("variadicGenericLambdaKernelIsWorking", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    std::uint32_t const arg1 = 42u;
    std::uint32_t const arg2 = 43u;
    VariadicGenericLambdaKernel<Acc> kernel;

    REQUIRE(fixture(kernel, arg1, arg2));
}

#endif
