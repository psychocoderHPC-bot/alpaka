/* Copyright 2025 Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/platform/PlatformCpu.hpp"

#include <span>

namespace aV2P::trait
{
    //! The std::span device type trait specialization.
    template<typename TElem>
    struct DevType<std::span<TElem>>
    {
        using type = DevCpu;
    };

    //! The std::span device get trait specialization.
    template<typename TElem>
    struct GetDev<std::span<TElem>>
    {
        AV2P_FN_HOST static auto getDev(std::span<TElem> const& /* view */) -> DevCpu
        {
            // Instantiating the CPU platform here is a hack we can do internally, because we know that the CPU
            // platform does not contain any data. But it generally does not apply.
            return getDevByIdx(PlatformCpu{}, 0u);
        }
    };

    //! The std::span dimension getter trait specialization.
    template<typename TElem>
    struct DimType<std::span<TElem>>
    {
        using type = DimInt<1u>;
    };

    //! The std::span memory element type get trait specialization.
    template<typename TElem>
    struct ElemType<std::span<TElem>>
    {
        using type = TElem;
    };

    template<typename TElem>
    struct GetExtents<std::span<TElem>>
    {
        AV2P_FN_HOST constexpr auto operator()(std::span<TElem> const& a) -> Vec<DimInt<1>, Idx<std::span<TElem>>>
        {
            return {std::size(a)};
        }
    };

    //! The std::span native pointer get trait specialization.
    template<typename TElem>
    struct GetPtrNative<std::span<TElem>>
    {
        AV2P_FN_HOST static auto getPtrNative(std::span<TElem> const& view) -> TElem const*
        {
            return std::data(view);
        }

        AV2P_FN_HOST static auto getPtrNative(std::span<TElem>& view) -> TElem*
        {
            return std::data(view);
        }
    };

    //! The std::span offset get trait specialization.
    template<typename TElem>
    struct GetOffsets<std::span<TElem>>
    {
        AV2P_FN_HOST auto operator()(std::span<TElem> const&) -> Vec<DimInt<1>, Idx<std::span<TElem>>>
        {
            return {0};
        }
    };

    //! The std::span idx type trait specialization.
    template<typename TElem>
    struct IdxType<std::span<TElem>>
    {
        using type = std::size_t;
    };
} // namespace aV2P::trait
