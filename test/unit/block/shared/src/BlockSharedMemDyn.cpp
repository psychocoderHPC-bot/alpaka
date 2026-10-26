/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/block/shared/dyn/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class BlockSharedMemDynTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Assure that the pointer is non null.
        auto a = aV2P::getDynSharedMem<std::uint32_t>(acc);
        AV2P_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != a);

        // Each call should return the same pointer ...
        auto b = aV2P::getDynSharedMem<std::uint32_t>(acc);
        AV2P_CHECK(*success, a == b);

        // ... even for different types.
        auto c = aV2P::getDynSharedMem<float>(acc);
        AV2P_CHECK(*success, a == reinterpret_cast<std::uint32_t*>(c));
    }
};

namespace aV2P::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<BlockSharedMemDynTestKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec>
        AV2P_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            BlockSharedMemDynTestKernel const& /* blockSharedMemDyn */,
            TVec const& blockThreadExtent,
            TVec const& threadElemExtent,
            bool* /* success */) -> std::size_t
        {
            auto const gridSize = blockThreadExtent.prod() * threadElemExtent.prod();
            return static_cast<std::size_t>(gridSize) * sizeof(std::uint32_t);
        }
    };
} // namespace aV2P::trait

TEMPLATE_LIST_TEST_CASE("sameNonNullAdress", "[blockSharedMemDyn]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    BlockSharedMemDynTestKernel kernel;

    REQUIRE(fixture(kernel));
}
