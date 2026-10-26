/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/intrinsic/IntrinsicFallback.hpp"
#include "aV2P/intrinsic/Traits.hpp"

#include <cstdint>

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P
{
    //! The SYCL intrinsic.
    class IntrinsicGenericSycl : public interface::Implements<ConceptIntrinsic, IntrinsicGenericSycl>
    {
    };
} // namespace aV2P

namespace aV2P::trait
{
    template<>
    struct Popcount<IntrinsicGenericSycl>
    {
        static auto popcount(IntrinsicGenericSycl const&, std::uint32_t value) -> std::int32_t
        {
            return static_cast<std::int32_t>(sycl::popcount(value));
        }

        static auto popcount(IntrinsicGenericSycl const&, std::uint64_t value) -> std::int32_t
        {
            return static_cast<std::int32_t>(sycl::popcount(value));
        }
    };

    template<>
    struct Ffs<IntrinsicGenericSycl>
    {
        static auto ffs(IntrinsicGenericSycl const&, std::int32_t value) -> std::int32_t
        {
            // There is no FFS operation in SYCL but we can emulate it using popcount.
            return (value == 0) ? 0 : sycl::popcount(value ^ ~(-value));
        }

        static auto ffs(IntrinsicGenericSycl const&, std::int64_t value) -> std::int32_t
        {
            // There is no FFS operation in SYCL but we can emulate it using popcount.
            return (value == 0l) ? 0 : static_cast<std::int32_t>(sycl::popcount(value ^ ~(-value)));
        }
    };
} // namespace aV2P::trait

#endif
