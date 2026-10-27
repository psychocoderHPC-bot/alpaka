/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/intrinsic/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

template<typename TInput>
class PopcountTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Use negative values to get inputs near the max value of TInput type
        TInput const inputs[]
            = {0u,
               1u,
               3u,
               54u,
               163u,
               51362u,
               static_cast<TInput>(-43631),
               static_cast<TInput>(-1352),
               static_cast<TInput>(-642),
               static_cast<TInput>(-1)};
        for(auto const input : inputs)
        {
            int const expected = popcountNaive(input);
            int const actual = aV2P::popcount(acc, input);
            AV2P_CHECK(*success, actual == expected);
        }
    }

private:
    AV2P_FN_ACC static auto popcountNaive(TInput value) -> int
    {
        int result = 0;
        while(value)
        {
            result += static_cast<int>(value & 1u);
            value >>= 1u;
        }
        return result;
    }
};

TEMPLATE_LIST_TEST_CASE("popcount", "[intrinsic]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    PopcountTestKernel<std::uint32_t> kernel32bit;
    REQUIRE(fixture(kernel32bit));

    PopcountTestKernel<std::uint64_t> kernel64bit;
    REQUIRE(fixture(kernel64bit));
}
