/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/rand/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

class RandTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    AV2P_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            auto dist = aV2P::rand::distribution::createNormalReal<float>(acc);
            [[maybe_unused]] auto const r = dist(gen);
            if constexpr(!AV2P_ARCH_PTX)
                AV2P_CHECK(*success, std::isfinite(r));
        }

        {
            auto dist = aV2P::rand::distribution::createNormalReal<double>(acc);
            [[maybe_unused]] auto const r = dist(gen);
            if constexpr(!AV2P_ARCH_PTX)
                AV2P_CHECK(*success, std::isfinite(r));
        }
        {
            auto dist = aV2P::rand::distribution::createUniformReal<float>(acc);
            auto const r = dist(gen);
            AV2P_CHECK(*success, 0.0f <= r);
            AV2P_CHECK(*success, 1.0f > r);
        }

        {
            auto dist = aV2P::rand::distribution::createUniformReal<double>(acc);
            auto const r = dist(gen);
            AV2P_CHECK(*success, 0.0 <= r);
            AV2P_CHECK(*success, 1.0 > r);
        }

        {
            auto dist = aV2P::rand::distribution::createUniformUint<std::uint32_t>(acc);
            [[maybe_unused]] auto const r = dist(gen);
        }
    }

public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // default generator for accelerator
        auto genDefault = aV2P::rand::engine::createDefault(acc, 12345u, 6789u);
        genNumbers(acc, success, genDefault);

#if !defined(AV2P_ACC_GPU_CUDA_ENABLED) && !defined(AV2P_ACC_GPU_HIP_ENABLED) && !defined(AV2P_ACC_SYCL_ENABLED)
        // TODO: These ifdefs are wrong: They will reduce the test to the
        // smallest common denominator from all enabled backends
        // std::random_device
        auto genRandomDevice = aV2P::rand::engine::createDefault(aV2P::rand::RandomDevice{}, 12345u, 6789u);
        genNumbers(acc, success, genRandomDevice);

        // MersenneTwister
        auto genMersenneTwister = aV2P::rand::engine::createDefault(aV2P::rand::MersenneTwister{}, 12345u, 6789u);
        genNumbers(acc, success, genMersenneTwister);

        // TinyMersenneTwister
        auto genTinyMersenneTwister
            = aV2P::rand::engine::createDefault(aV2P::rand::TinyMersenneTwister{}, 12345u, 6789u);
        genNumbers(acc, success, genTinyMersenneTwister);
#endif
    }
};

TEMPLATE_LIST_TEST_CASE("defaultRandomGeneratorIsWorking", "[rand]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    RandTestKernel kernel;

    REQUIRE(fixture(kernel));
}

//! Helper trait to check if the given accelerator is HIP
template<typename TAcc>
struct IsAccHIP : public std::false_type
{
};

#if defined(AV2P_ACC_GPU_HIP_ENABLED)
template<typename TDim, typename TIdx>
struct IsAccHIP<aV2P::AccGpuHipRt<TDim, TIdx>> : public std::true_type
{
};
#endif

TEMPLATE_LIST_TEST_CASE("defaultRandomGeneratorIsTriviallyCopyable", "[rand]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using DefaultEngine = decltype(aV2P::rand::engine::createDefault(std::declval<Acc>(), 0u, 0u, 0u));
    STATIC_REQUIRE(std::is_trivially_copyable_v<DefaultEngine>);
}
