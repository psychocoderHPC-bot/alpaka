/* Copyright 2020 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/block/sync/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class BlockSyncPredicateTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        using Idx = aV2P::Idx<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx = aV2P::getIdx<aV2P::Block, aV2P::Threads>(acc);
        auto const blockThreadExtent = aV2P::getWorkDiv<aV2P::Block, aV2P::Threads>(acc);
        auto const blockThreadIdx1D = aV2P::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockThreadExtent1D = blockThreadExtent.prod();

        // syncBlockThreadsPredicate<aV2P::BlockCount>
        {
            Idx const modulus = 2u;
            int const predicate = static_cast<int>(blockThreadIdx1D % modulus);
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockCount>(acc, predicate);
            auto const expectedResult = static_cast<int>(blockThreadExtent1D / modulus);
            AV2P_CHECK(*success, expectedResult == result);
        }
        {
            Idx const modulus = 3u;
            int const predicate = static_cast<int>(blockThreadIdx1D % modulus);
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockCount>(acc, predicate);
            auto const expectedResult = static_cast<int>(
                blockThreadExtent1D - ((blockThreadExtent1D + modulus - static_cast<Idx>(1u)) / modulus));
            AV2P_CHECK(*success, expectedResult == result);
        }

        // syncBlockThreadsPredicate<aV2P::BlockAnd>
        {
            int const predicate = 1;
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockAnd>(acc, predicate);
            AV2P_CHECK(*success, result == 1);
        }
        {
            int const predicate = 0;
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockAnd>(acc, predicate);
            AV2P_CHECK(*success, result == 0);
        }
        {
            int const predicate = blockThreadIdx1D != 0;
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockAnd>(acc, predicate);
            AV2P_CHECK(*success, result == 0);
        }

        // syncBlockThreadsPredicate<aV2P::BlockOr>
        {
            int const predicate = 1;
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockOr>(acc, predicate);
            AV2P_CHECK(*success, result == 1);
        }
        {
            int const predicate = 0;
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockOr>(acc, predicate);
            AV2P_CHECK(*success, result == 0);
        }
        {
            int const predicate = static_cast<int>(blockThreadIdx1D != 1);
            auto const result = aV2P::syncBlockThreadsPredicate<aV2P::BlockOr>(acc, predicate);
            AV2P_CHECK(*success, result == 1);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("synchronizePredicate", "[blockSync]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    BlockSyncPredicateTestKernel kernel;

    // 4^Dim
    {
        aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::all(static_cast<Idx>(4u)));

        REQUIRE(fixture(kernel));
    }

    // 1^Dim
    {
        aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

        REQUIRE(fixture(kernel));
    }
}
