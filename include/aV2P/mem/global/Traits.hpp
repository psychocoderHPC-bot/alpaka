/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/meta/DependentFalseType.hpp"

namespace aV2P
{

    namespace detail
    {
        template<concepts::Tag TTag, typename T>
        struct DevGlobalImplGeneric
        {
            // does not make use of TTag
            using Type = std::remove_const_t<T>;
            Type value; // backend specific value

            AV2P_FN_HOST_ACC T* operator&()
            {
                return &value;
            }

            AV2P_FN_HOST_ACC T& get()
            {
                return value;
            }
        };

        template<concepts::Tag TTag, typename T>
        struct DevGlobalTrait
        {
            static constexpr bool const IsImplementedFor = aV2P::meta::DependentFalseType<TTag>::value;

            static_assert(IsImplementedFor, "Error: device global variables are not implemented for the given Tag");
        };
    } // namespace detail

    template<typename TAcc, typename T>
    using DevGlobal = typename detail::DevGlobalTrait<typename aV2P::trait::AccToTag<TAcc>::type, T>::Type;
} // namespace aV2P
