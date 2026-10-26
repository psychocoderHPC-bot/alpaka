/* Copyright 2025 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci,
 * Aurora Perego SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/Common.hpp>
#include <aV2P/core/Config.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using Elem = std::uint32_t;
using Dim = aV2P::DimInt<2u>;
using Idx = std::uint32_t;

AV2P_STATIC_ACC_MEM_GLOBAL aV2P::DevGlobal<TAcc, Elem[3][2]> g_globalMemory2DUninitialized;

AV2P_STATIC_ACC_MEM_CONSTANT aV2P::DevGlobal<TAcc, const Elem[3][2]> g_constantMemory2DUninitialized;

//! Uses static device memory on the accelerator defined globally for the whole compilation unit.
struct StaticDeviceMemoryTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    // AV2P_FN_ACC void operator()(TAcc const& acc, bool* success) const
    AV2P_FN_ACC void operator()(TAcc const& acc, bool* success) const
    {
        auto const gridThreadExtent = aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc);
        auto const gridThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc);

        auto const offset = gridThreadExtent[1u] * gridThreadIdx[0u] + gridThreadIdx[1u];
        auto const val = offset;

        AV2P_CHECK(*success, val == *((&g_globalMemory2DUninitialized<TAcc>.get())[0][0] + offset));
    }
};

struct ConstantDeviceMemoryTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& acc, bool* success) const
    {
        auto const gridThreadExtent = aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc);
        auto const gridThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc);

        auto const offset = gridThreadExtent[1u] * gridThreadIdx[0u] + gridThreadIdx[1u];
        auto const val = offset;

        AV2P_CHECK(*success, val == *((&g_constantMemory2DUninitialized<TAcc>.get())[0][0] + offset));
    }
};

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_CPU)
template<typename TDim, typename TIdx>
using EnabledAccsElseInt = std::tuple<aV2P::AccCpuSycl<TDim, TIdx>>;
template<typename TDim, typename TIdx>
using EnabledAccs = typename aV2P::meta::Filter<EnabledAccsElseInt<TDim, TIdx>, std::is_class>;
using TestAccs = EnabledAccs<Dim, Idx>;
#elif defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_GPU)
template<typename TDim, typename TIdx>
using EnabledAccsElseInt = std::tuple<aV2P::AccGpuSyclIntel<TDim, TIdx>>;
template<typename TDim, typename TIdx>
using EnabledAccs = typename aV2P::meta::Filter<EnabledAccsElseInt<TDim, TIdx>, std::is_class>;
using TestAccs = EnabledAccs<Dim, Idx>;
#elif defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_GPU_NVIDIA)
template<typename TDim, typename TIdx>
using EnabledAccsElseInt = std::tuple<aV2P::AccGpuSyclNvidia<TDim, TIdx>>;
template<typename TDim, typename TIdx>
using EnabledAccs = typename aV2P::meta::Filter<EnabledAccsElseInt<TDim, TIdx>, std::is_class>;
using TestAccs = EnabledAccs<Dim, Idx>;
#elif defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_GPU_AMD)
template<typename TDim, typename TIdx>
using EnabledAccsElseInt = std::tuple<aV2P::AccGpuSyclAmd<TDim, TIdx>>;
template<typename TDim, typename TIdx>
using EnabledAccs = typename aV2P::meta::Filter<EnabledAccsElseInt<TDim, TIdx>, std::is_class>;
using TestAccs = EnabledAccs<Dim, Idx>;
#else
using TestAccs = aV2P::test::EnabledAccs<Dim, Idx>;
#endif

TEMPLATE_LIST_TEST_CASE("staticDeviceMemoryGlobal", "[viewStaticAccMem]", TestAccs)
{
    using Acc = TestType;
    using DevAcc = aV2P::Dev<Acc>;

    auto const platformAcc = aV2P::Platform<Acc>{};

    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    aV2P::Vec<Dim, Idx> const extent(3u, 2u);

    // uninitialized static global device memory
    {
        auto const platformHost = aV2P::PlatformCpu{};
        auto const devHost = aV2P::getDevByIdx(platformHost, 0);

        using QueueAcc = aV2P::test::DefaultQueue<DevAcc>;
        QueueAcc queueAcc(devAcc);

        aV2P::test::KernelExecutionFixture<Acc> fixture(queueAcc, extent);

        StaticDeviceMemoryTestKernel kernel;

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        auto bufHost = aV2P::createView(devHost, data.data(), extent);

        aV2P::memcpy(queueAcc, g_globalMemory2DUninitialized<Acc>, bufHost, extent);
        aV2P::wait(queueAcc);

        REQUIRE(fixture(kernel));

        std::vector<Elem> data2(6, 0u);
        auto bufHost2 = aV2P::createView(devHost, data2.data(), extent);
        aV2P::memcpy(queueAcc, bufHost2, g_globalMemory2DUninitialized<Acc>, extent);
        aV2P::wait(queueAcc);
        REQUIRE(data == data2);
    }
}

TEMPLATE_LIST_TEST_CASE("staticDeviceMemoryConstant", "[viewStaticAccMem]", TestAccs)
{
    using Acc = TestType;
    using DevAcc = aV2P::Dev<Acc>;

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    aV2P::Vec<Dim, Idx> const extent(3u, 2u);

    // uninitialized static constant device memory
    {
        auto const platformHost = aV2P::PlatformCpu{};
        auto const devHost = aV2P::getDevByIdx(platformHost, 0);

        using QueueAcc = aV2P::test::DefaultQueue<DevAcc>;
        QueueAcc queueAcc(devAcc);

        aV2P::test::KernelExecutionFixture<Acc> fixture(queueAcc, extent);

        ConstantDeviceMemoryTestKernel kernel;

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        auto bufHost = aV2P::createView(devHost, data.data(), extent);

        aV2P::memcpy(queueAcc, g_constantMemory2DUninitialized<Acc>, bufHost);
        aV2P::wait(queueAcc);

        REQUIRE(fixture(kernel));

        std::vector<Elem> data2(6, 0u);
        auto bufHost2 = aV2P::createView(devHost, data2.data(), extent);
        aV2P::memcpy(queueAcc, bufHost2, g_constantMemory2DUninitialized<Acc>);
        aV2P::wait(queueAcc);
        REQUIRE(data == data2);
    }
}
