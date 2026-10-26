/* Copyright 2025 Simeon Ehrig, Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

// Undefine AV2P_CI for this test, because the variable set the value of some acc types, like
// AccCpuThreadsIfAvailableElseInt to int independent of the cmake configuration. This avoids long running test cases
// but is problematic for this test.
#undef AV2P_CI
#include <aV2P/alpaka.hpp>
#include <aV2P/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

using Dim = aV2P::DimInt<1>;
using Idx = int;
using TestAccs = aV2P::test::EnabledAccs<Dim, Idx>;

using TagList = std::tuple<
    aV2P::TagCpuSerial,
    aV2P::TagCpuThreads,
    aV2P::TagCpuTbbBlocks,
    aV2P::TagCpuOmp2Blocks,
    aV2P::TagCpuOmp2Threads,
    aV2P::TagGpuCudaRt,
    aV2P::TagGpuHipRt,
    aV2P::TagCpuSycl,
    aV2P::TagGpuSyclIntel,
    aV2P::TagGpuSyclNvidia,
    aV2P::TagGpuSyclAmd>;

using AccToTagMap = std::tuple<
    std::pair<aV2P::test::detail::AccCpuSerialIfAvailableElseInt<Dim, Idx>, aV2P::TagCpuSerial>,
    std::pair<aV2P::test::detail::AccCpuThreadsIfAvailableElseInt<Dim, Idx>, aV2P::TagCpuThreads>,
    std::pair<aV2P::test::detail::AccCpuTbbIfAvailableElseInt<Dim, Idx>, aV2P::TagCpuTbbBlocks>,
    std::pair<aV2P::test::detail::AccCpuOmp2BlocksIfAvailableElseInt<Dim, Idx>, aV2P::TagCpuOmp2Blocks>,
    std::pair<aV2P::test::detail::AccCpuOmp2ThreadsIfAvailableElseInt<Dim, Idx>, aV2P::TagCpuOmp2Threads>,
    std::pair<aV2P::test::detail::AccGpuCudaRtIfAvailableElseInt<Dim, Idx>, aV2P::TagGpuCudaRt>,
    std::pair<aV2P::test::detail::AccGpuHipRtIfAvailableElseInt<Dim, Idx>, aV2P::TagGpuHipRt>,
    std::pair<aV2P::test::detail::AccCpuSyclIfAvailableElseInt<Dim, Idx>, aV2P::TagCpuSycl>,
    std::pair<aV2P::test::detail::AccGpuSyclIntelIfAvailableElseInt<Dim, Idx>, aV2P::TagGpuSyclIntel>,
    std::pair<aV2P::test::detail::AccGpuSyclNvidiaIfAvailableElseInt<Dim, Idx>, aV2P::TagGpuSyclNvidia>,
    std::pair<aV2P::test::detail::AccGpuSyclAmdIfAvailableElseInt<Dim, Idx>, aV2P::TagGpuSyclAmd>>;

using AccTagTestMatrix = aV2P::meta::CartesianProduct<std::tuple, AccToTagMap, TagList>;

TEMPLATE_LIST_TEST_CASE("test all possible acc tag combinations", "[acc][tag]", AccTagTestMatrix)
{
    // type of TestType is std::tuple<std::tuple<TAcc, TTag>, TTag>
    // TAcc is ether a alpaka accelerator or int, if not enabled
    // The Tag in in the inner tuple is the expected Tag of the TAcc
    // The Tag in the outer tuple is the Tag to test
    using TestAccTestTuple = std::tuple_element_t<0, TestType>;
    using TestAcc = std::tuple_element_t<0, TestAccTestTuple>;
    using ExpectedTag = std::tuple_element_t<1, TestAccTestTuple>;
    using TestTag = std::tuple_element_t<1, TestType>;

    // if the Acc is not enabled, the type is int
    if constexpr(!std::is_same_v<TestAcc, int>)
    {
        if constexpr(std::is_same_v<TestTag, ExpectedTag>)
        {
            STATIC_REQUIRE((std::is_same_v<aV2P::AccToTag<TestAcc>, ExpectedTag>) );
            STATIC_REQUIRE((std::is_same_v<aV2P::TagToAcc<TestTag, Dim, Idx>, TestAcc>) );
            STATIC_REQUIRE((aV2P::accMatchesTags<TestAcc, TestTag>) );
        }
        else
        {
            STATIC_REQUIRE(!(std::is_same_v<aV2P::AccToTag<TestAcc>, TestTag>) );
            STATIC_REQUIRE(!(aV2P::accMatchesTags<TestAcc, TestTag>) );
        }
    }
}

// ###################################################
// first version of function specialization
// ###################################################

template<typename Tag>
std::string specialized_function()
{
    return "Generic";
}

template<>
std::string specialized_function<aV2P::TagCpuSerial>()
{
    return "Serial";
}

template<>
std::string specialized_function<aV2P::TagGpuCudaRt>()
{
    return "CUDA";
}

// ###################################################
// second version of function specialization
// ###################################################

template<typename TTag>
std::string specialized_function_2(TTag)
{
    return "Generic";
}

// is required because of -Werror=missing-declarations
std::string specialized_function_2(aV2P::TagCpuSerial);

std::string specialized_function_2(aV2P::TagCpuSerial)
{
    return "Serial";
}

std::string specialized_function_2(aV2P::TagGpuCudaRt);

std::string specialized_function_2(aV2P::TagGpuCudaRt)
{
    return "CUDA";
}

TEMPLATE_LIST_TEST_CASE("specialization of functions with tags", "[acc][tag]", TestAccs)
{
    using TestAcc = TestType;

    std::string expected_result = "Generic";

#if defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    if constexpr(std::is_same_v<aV2P::AccCpuSerial<Dim, Idx>, TestAcc>)
    {
        expected_result = "Serial";
    }
#endif

#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<aV2P::AccGpuCudaRt<Dim, Idx>, TestAcc>)
    {
        expected_result = "CUDA";
    }
#endif

    INFO(
        "Test Acc: " + aV2P::getAccName<TestAcc>() + "\n" + "expect: " + expected_result + "\n"
        + "is_specialized() returns: " + specialized_function<aV2P::AccToTag<TestAcc>>() + "\n"
        + "is_specialized_2() returns: " + specialized_function_2(aV2P::AccToTag<TestAcc>{}));
    REQUIRE((specialized_function<aV2P::AccToTag<TestAcc>>() == expected_result));
    REQUIRE((specialized_function_2(aV2P::AccToTag<TestAcc>{}) == expected_result));
}

struct InitKernel
{
    template<typename TAcc, typename TData>
    AV2P_FN_ACC auto operator()(TAcc const&, TData* const data) const noexcept -> void
    {
        data[0] = 42;
    }
};

template<typename TAcc, typename = void>
struct specialized_Kernel
{
    template<typename TData>
    AV2P_FN_INLINE AV2P_FN_ACC TData operator()(TAcc const& acc, TData input)
    {
        return aV2P::math::min(acc, 0, input);
    }
};

template<typename TAcc>
struct specialized_Kernel<TAcc, std::enable_if_t<aV2P::accMatchesTags<TAcc, aV2P::TagGpuCudaRt>>>
{
    template<typename TData>
    AV2P_FN_INLINE AV2P_FN_ACC TData operator()(TAcc const& acc, TData input)
    {
        return aV2P::math::min(acc, 1, input);
    }
};

template<typename TAcc>
struct specialized_Kernel<
    TAcc,
    std::enable_if_t<aV2P::accMatchesTags<TAcc, aV2P::TagCpuOmp2Blocks, aV2P::TagCpuOmp2Threads>>>
{
    template<typename TData>
    AV2P_FN_INLINE AV2P_FN_ACC TData operator()(TAcc const& acc, TData input)
    {
        return aV2P::math::min(acc, 2, input);
    }
};

struct WrapperKernel
{
    template<typename TAcc, typename TData>
    AV2P_FN_ACC auto operator()(TAcc const& acc, TData* const data) const noexcept -> void
    {
        data[0] = specialized_Kernel<TAcc>{}(acc, data[0]);
    }
};

TEMPLATE_LIST_TEST_CASE("kernel specialization with tags", "[acc][tag]", TestAccs)
{
    using TestAcc = TestType;
    using QueueProperty = aV2P::Blocking;
    using Queue = aV2P::Queue<TestAcc, QueueProperty>;
    auto const platformAcc = aV2P::Platform<TestAcc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);
    Queue queue(devAcc);

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    using Vec = aV2P::Vec<aV2P::DimInt<1>, int>;
    Vec extents(Vec::all(1));

    auto memHost = aV2P::allocBuf<int, Idx>(devHost, extents);
    auto memAcc = aV2P::allocBuf<int, Idx>(devAcc, extents);
    int* const memAccPtr = aV2P::getPtrNative(memAcc);

    aV2P::WorkDivMembers<Dim, Idx> workdiv{1, 1, 1};

    aV2P::exec<TestAcc>(queue, workdiv, InitKernel{}, memAccPtr);
    aV2P::exec<TestAcc>(queue, workdiv, WrapperKernel{}, memAccPtr);
    aV2P::wait(queue);

    aV2P::memcpy(queue, memHost, memAcc);
    aV2P::wait(queue);

    // result of the generic implementation
    int expected_result = 0;

#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<aV2P::AccGpuCudaRt<Dim, Idx>, TestAcc>)
    {
        expected_result = 1;
    }
#endif

#if defined(AV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    if constexpr(std::is_same_v<aV2P::AccCpuOmp2Threads<Dim, Idx>, TestAcc>)
    {
        expected_result = 2;
    }
#endif

#if defined(AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    if constexpr(std::is_same_v<aV2P::AccCpuOmp2Blocks<Dim, Idx>, TestAcc>)
    {
        expected_result = 2;
    }
#endif

    REQUIRE(aV2P::getPtrNative(memHost)[0] == expected_result);
}

TEMPLATE_LIST_TEST_CASE("test AccIsEnabled", "[acc][tag]", AccToTagMap)
{
    using TestAcc = std::tuple_element_t<0, TestType>;
    using TestTag = std::tuple_element_t<1, TestType>;


    // if the Acc is not enabled, the type is int
    if constexpr(!std::is_same_v<TestAcc, int>)
    {
        STATIC_REQUIRE(aV2P::AccIsEnabled<TestTag>::value);
    }
    else
    {
        STATIC_REQUIRE_FALSE(aV2P::AccIsEnabled<TestTag>::value);
    }
}

TEST_CASE("test EnabledAccTags", "[acc][tag]")
{
    using AllAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1>, int>;
    STATIC_REQUIRE(std::tuple_size<AllAccs>::value == std::tuple_size<aV2P::EnabledAccTags>::value);
}

struct NoTag1
{
    static std::string get_name()
    {
        return "foo";
    }
};

struct NoTag2 : public aV2P::InterfaceTag
{
};

template<aV2P::concepts::Tag>
consteval bool specialize_tag()
{
    return true;
}

TEMPLATE_LIST_TEST_CASE("test concept tag", "[tag][concept]", TagList)
{
    using TagToTest = TestType;
    STATIC_REQUIRE(aV2P::concepts::Tag<TagToTest>);
    STATIC_REQUIRE_FALSE(aV2P::concepts::Tag<NoTag1>);
    STATIC_REQUIRE_FALSE(aV2P::concepts::Tag<NoTag2>);

    STATIC_REQUIRE(specialize_tag<TagToTest>());
}
