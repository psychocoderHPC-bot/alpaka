/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/alpaka.hpp"

#include <type_traits>

namespace aV2P::test
{
    namespace trait
    {
        // \tparam T Type to conditionally make const.
        // \tparam TSource Type to mimic the constness of.
        template<typename T, typename TSource>
        using MimicConst = std::conditional_t<std::is_const_v<TSource>, std::add_const_t<T>, std::remove_const_t<T>>;

        template<typename TView, typename TSfinae = void>
        class IteratorView
        {
            using TViewDecayed = std::decay_t<TView>;
            using Dim = aV2P::Dim<TViewDecayed>;
            using Idx = aV2P::Idx<TViewDecayed>;
            using Elem = MimicConst<aV2P::Elem<TViewDecayed>, TView>;

        public:
            AV2P_FN_HOST IteratorView(TView& view, Idx const idx)
                : m_nativePtr(getPtrNative(view))
                , m_currentIdx(idx)
                , m_extents(getExtents(view))
                , m_pitchBytes(getPitchesInBytes(view))
            {
            }

            AV2P_FN_HOST explicit IteratorView(TView& view) : IteratorView(view, 0)
            {
            }

            AV2P_FN_HOST_ACC auto operator++() -> IteratorView&
            {
                ++m_currentIdx;
                return *this;
            }

            AV2P_FN_HOST_ACC auto operator--() -> IteratorView&
            {
                --m_currentIdx;
                return *this;
            }

            AV2P_FN_HOST_ACC auto operator++(int) -> IteratorView
            {
                IteratorView iterCopy = *this;
                m_currentIdx++;
                return iterCopy;
            }

            AV2P_FN_HOST_ACC auto operator--(int) -> IteratorView
            {
                IteratorView iterCopy = *this;
                m_currentIdx--;
                return iterCopy;
            }

            template<typename TIter>
            AV2P_FN_HOST_ACC auto operator==(TIter& other) const -> bool
            {
                return m_currentIdx == other.m_currentIdx;
            }

            template<typename TIter>
            AV2P_FN_HOST_ACC auto operator!=(TIter& other) const -> bool
            {
                return m_currentIdx != other.m_currentIdx;
            }

            AV2P_FN_HOST_ACC auto operator*() const -> Elem&
            {
                if constexpr(Dim::value == 0)
                    return *m_nativePtr;
                else
                {
                    Vec<Dim, Idx> const currentIdxDimx
                        = mapIdx<Dim::value>(Vec<DimInt<1>, Idx>{m_currentIdx}, m_extents);
                    auto const offsetInBytes = (currentIdxDimx * m_pitchBytes).sum();
                    using QualifiedByte = MimicConst<std::byte, Elem>;
                    return *reinterpret_cast<Elem*>(__builtin_assume_aligned(
                        reinterpret_cast<QualifiedByte*>(m_nativePtr) + offsetInBytes,
                        alignof(Elem)));
                }
                AV2P_UNREACHABLE(*m_nativePtr);
            }

        private:
            Elem* m_nativePtr;
            Idx m_currentIdx;
            Vec<Dim, Idx> m_extents;
            Vec<Dim, Idx> m_pitchBytes;
        };

        template<typename TView, typename TSfinae = void>
        struct Begin
        {
            AV2P_FN_HOST static auto begin(TView& view) -> IteratorView<TView>
            {
                return IteratorView<TView>(view);
            }
        };

        template<typename TView, typename TSfinae = void>
        struct End
        {
            AV2P_FN_HOST static auto end(TView& view) -> IteratorView<TView>
            {
                auto extents = getExtents(view);
                return IteratorView<TView>(view, extents.prod());
            }
        };
    } // namespace trait

    template<typename TView>
    using Iterator = trait::IteratorView<TView>;

    template<typename TView>
    AV2P_FN_HOST auto begin(TView& view) -> Iterator<TView>
    {
        return trait::Begin<TView>::begin(view);
    }

    template<typename TView>
    AV2P_FN_HOST auto end(TView& view) -> Iterator<TView>
    {
        return trait::End<TView>::end(view);
    }
} // namespace aV2P::test
