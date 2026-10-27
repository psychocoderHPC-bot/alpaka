/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace aV2P::adapter
{
    //! aV2P-owned device specification used as the template argument of all adapter wrappers.
    //!
    //! Using an aV2P type instead of `alpaka::onHost::DeviceSpec` as a template argument keeps
    //! `alpaka::onHost` out of the argument-dependent lookup set of unrelated aV2P templates.
    //! Translation to the alpaka 3 type happens in `toAlpaka()`.
    template<alpaka::concepts::Api TApi, alpaka::concepts::DeviceKind TDeviceKind>
    struct DeviceSpec
    {
        using Api = TApi;
        using DeviceKind = TDeviceKind;

        static constexpr auto toAlpaka()
        {
            return alpaka::onHost::DeviceSpec<TApi, TDeviceKind>{};
        }
    };
} // namespace aV2P::adapter
