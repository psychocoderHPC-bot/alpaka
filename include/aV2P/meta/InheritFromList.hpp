/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace aV2P::meta
{
    template<typename TBaseList>
    class InheritFromList;

    template<template<typename...> class TList, typename... TBases>
    class InheritFromList<TList<TBases...>> : public TBases...
    {
    };
} // namespace aV2P::meta
