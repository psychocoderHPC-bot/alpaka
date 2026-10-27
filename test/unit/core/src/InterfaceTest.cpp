/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/Interface.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

struct InterfaceExample
{
};

struct InterfaceNonMatchingExample
{
};

struct ImplementerNotTagged
{
};

struct ImplementerNotTaggedButNonMatchingTagged
    : public aV2P::interface::Implements<InterfaceNonMatchingExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerTagged : public aV2P::interface::Implements<InterfaceExample, ImplementerTagged>
{
};

struct ImplementerTaggedButAlsoNonMatchingTagged
    : public aV2P::interface::Implements<InterfaceNonMatchingExample, ImplementerTaggedButAlsoNonMatchingTagged>
    , public aV2P::interface::Implements<InterfaceExample, ImplementerTaggedButAlsoNonMatchingTagged>
{
};

struct ImplementerWithTaggedBase : public ImplementerTagged
{
};

struct ImplementerWithTaggedBaseAlsoNonMatchingTagged : public ImplementerTaggedButAlsoNonMatchingTagged
{
};

struct ImplementerTaggedToBase
    : public ImplementerNotTagged
    , public aV2P::interface::Implements<InterfaceExample, ImplementerNotTagged>
{
};

struct ImplementerTaggedToBaseAlsoNonMatchingTagged
    : public ImplementerNotTaggedButNonMatchingTagged
    , public aV2P::interface::Implements<InterfaceExample, ImplementerNotTaggedButNonMatchingTagged>
{
};

struct ImplementerNonMatchingTaggedTaggedToBase
    : public ImplementerNotTagged
    , public aV2P::interface::Implements<InterfaceNonMatchingExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>
    , public aV2P::interface::Implements<InterfaceExample, ImplementerNotTagged>
{
};

TEST_CASE("ImplementerNotTagged", "[core]")
{
    using ImplementationBase = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerNotTagged>;

    static_assert(
        std::is_same_v<ImplementerNotTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerNotTaggedButNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerNotTaggedButNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerNotTaggedButNonMatchingTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTagged", "[core]")
{
    using ImplementationBase = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerTagged>;

    static_assert(
        std::is_same_v<ImplementerTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTaggedButAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerTaggedButAlsoNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerTaggedButAlsoNonMatchingTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerWithTaggedBaseAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerWithTaggedBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerTaggedButAlsoNonMatchingTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerWithTaggedBase", "[core]")
{
    using ImplementationBase = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerWithTaggedBase>;

    static_assert(
        std::is_same_v<ImplementerTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTaggedToBase", "[core]")
{
    using ImplementationBase = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerTaggedToBase>;

    static_assert(
        std::is_same_v<ImplementerNotTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerTaggedToBaseAlsoNonMatchingTagged", "[core]")
{
    using ImplementationBase
        = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerTaggedToBaseAlsoNonMatchingTagged>;

    static_assert(
        std::is_same_v<ImplementerNotTaggedButNonMatchingTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}

TEST_CASE("ImplementerNonMatchingTaggedTaggedToBase", "[core]")
{
    using ImplementationBase
        = aV2P::interface::ImplementationBase<InterfaceExample, ImplementerNonMatchingTaggedTaggedToBase>;

    static_assert(
        std::is_same_v<ImplementerNotTagged, ImplementationBase>,
        "aV2P::interface::ImplementationBase failed!");
}
