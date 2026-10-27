/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace aV2P::meta
{
    namespace detail
    {
        template<typename TList, template<typename...> class TApplicant>
        struct ApplyImpl;

        template<template<typename...> class TList, template<typename...> class TApplicant, typename... T>
        struct ApplyImpl<TList<T...>, TApplicant>
        {
            using type = TApplicant<T...>;
        };
    } // namespace detail
    template<typename TList, template<typename...> class TApplicant>
    using Apply = typename detail::ApplyImpl<TList, TApplicant>::type;
} // namespace aV2P::meta
