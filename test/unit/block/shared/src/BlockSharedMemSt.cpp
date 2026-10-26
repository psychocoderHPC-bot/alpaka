/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/block/shared/st/Traits.hpp>
#include <aV2P/test/Array.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class BlockSharedMemStNonNullTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
#if AV2P_COMP_GNUC >= AV2P_VERSION_NUMBER(6, 0, 0)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Waddress" // warning: the compiler can assume that the address of 'a' will never be NULL [-Waddress]
#endif
        // Multiple runs to make sure it really works.
        for(std::size_t i = 0u; i < 10; ++i)
        {
            auto& a = aV2P::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &a);

            auto& b = aV2P::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &b);

            auto& c = aV2P::declareSharedVar<float, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<float*>(nullptr) != &c);

            auto& d = aV2P::declareSharedVar<double, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<double*>(nullptr) != &d);

            auto& e = aV2P::declareSharedVar<std::uint64_t, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<std::uint64_t*>(nullptr) != &e);


            auto& f = aV2P::declareSharedVar<aV2P::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &f[0]);

            auto& g = aV2P::declareSharedVar<aV2P::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &g[0]);

            auto& h = aV2P::declareSharedVar<aV2P::test::Array<double, 16>, __COUNTER__>(acc);
            AV2P_CHECK(*success, static_cast<double*>(nullptr) != &h[0]);
        }
#if AV2P_COMP_GNUC >= AV2P_VERSION_NUMBER(6, 0, 0)
#    pragma GCC diagnostic pop
#endif
    }
};

TEMPLATE_LIST_TEST_CASE("nonNull", "[blockSharedMemSt]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    // Use multiple threads to make sure the synchronization really works.
    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::all(static_cast<Idx>(3u)));

    BlockSharedMemStNonNullTestKernel kernel;

    REQUIRE(fixture(kernel));
}

class BlockSharedMemStSameTypeAdressVerificationTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Allocations in the loop with same template signature should be equal too this allocation.
        // 21 byte is chosen to break nice alignment for all following calls of declareSharedVar.
        auto& baseAllocation = aV2P::declareSharedVar<aV2P::test::Array<std::uint8_t, 21>, 42>(acc);

        // Multiple runs to make sure it really works.
        for(std::size_t i = 0u; i < 10; ++i)
        {
            // same type but different id should result in different pointer for each allocation
            auto& a = aV2P::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            auto& b = aV2P::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            AV2P_CHECK(*success, &a != &b);
            auto& c = aV2P::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            AV2P_CHECK(*success, &b != &c);
            AV2P_CHECK(*success, &a != &c);
            AV2P_CHECK(*success, &b != &c);

            auto& d = aV2P::declareSharedVar<aV2P::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            AV2P_CHECK(*success, &a != &d[0]);
            AV2P_CHECK(*success, &b != &d[0]);
            AV2P_CHECK(*success, &c != &d[0]);
            auto& e = aV2P::declareSharedVar<aV2P::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            AV2P_CHECK(*success, &a != &e[0]);
            AV2P_CHECK(*success, &b != &e[0]);
            AV2P_CHECK(*success, &c != &e[0]);
            AV2P_CHECK(*success, &d[0] != &e[0]);
        }

        for(std::size_t i = 0u; i < 10; ++i)
        {
            // same type and same id should result in same pointer for each allocation
            auto& a = aV2P::declareSharedVar<aV2P::test::Array<std::uint8_t, 21>, 42>(acc);
            auto& b = aV2P::declareSharedVar<aV2P::test::Array<std::uint8_t, 21>, 42>(acc);
            AV2P_CHECK(*success, &a == &b);
            AV2P_CHECK(*success, &a == &baseAllocation);

            auto& lastAllocation = aV2P::declareSharedVar<aV2P::test::Array<std::uint8_t, 23>, 23>(acc);
            auto& c = aV2P::declareSharedVar<aV2P::test::Array<std::uint8_t, 23>, 23>(acc);
            AV2P_CHECK(*success, &lastAllocation == &c);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("sameTypeAddressVerification", "[blockSharedMemSt]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    // Use multiple threads to make sure the synchronization really works.
    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::all(static_cast<Idx>(3u)));

    BlockSharedMemStSameTypeAdressVerificationTestKernel kernel;

    REQUIRE(fixture(kernel));
}
