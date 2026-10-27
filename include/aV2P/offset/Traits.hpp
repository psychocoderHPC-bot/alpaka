/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

#include <type_traits>

namespace aV2P
{
    //! The offset traits.
    namespace trait
    {
        //! The x offset get trait.
        //!
        //! If not specialized explicitly it returns 0.
        template<typename TIdx, typename TOffsets, typename TSfinae = void>
        struct [[deprecated("Specialize GetOffsets instead")]] GetOffset
        {
            AV2P_NO_HOST_ACC_WARNING AV2P_FN_HOST_ACC static auto getOffset(TOffsets const&) -> Idx<TOffsets>
            {
                return static_cast<Idx<TOffsets>>(0);
            } // namespace trait
        }; // namespace aV2P

        //! The GetOffsets trait for getting the offsets of an object as an aV2P::Vec.
        template<typename TExtent, typename TSfinae = void>
        struct GetOffsets;
    } // namespace trait

    //! \return The offset in the given dimension.
    AV2P_NO_HOST_ACC_WARNING
    template<std::size_t Tidx, typename TOffsets>
    [[deprecated("use getOffsets(offsets)[Tidx] instead")]] AV2P_FN_HOST_ACC auto getOffset(TOffsets const& offsets)
        -> Idx<TOffsets>
    {
#if AV2P_COMP_CLANG || AV2P_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return trait::GetOffset<DimInt<Tidx>, TOffsets>::getOffset(offsets);
#if AV2P_COMP_CLANG || AV2P_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
    }

    //! \return The extents of the given object.
    AV2P_NO_HOST_ACC_WARNING
    template<typename T>
    AV2P_FN_HOST_ACC auto getOffsets(T const& object) -> Vec<Dim<T>, Idx<T>>
    {
        return trait::GetOffsets<T>{}(object);
    }

    //! \tparam T has to specialize GetOffsets.
    //! \return The offset vector.
    AV2P_NO_HOST_ACC_WARNING
    template<typename T>
    AV2P_FN_HOST_ACC constexpr auto getOffsetVec(T const& object = {}) -> Vec<Dim<T>, Idx<T>>
    {
        return getOffsets(object);
    }

    //! \tparam T has to specialize GetOffsets.
    //! \return The offset vector but only the last TDim elements.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TDim, typename T>
    AV2P_FN_HOST_ACC constexpr auto getOffsetVecEnd(T const& object = {}) -> Vec<TDim, Idx<T>>
    {
        static_assert(TDim::value <= Dim<T>::value, "Cannot get more items than the offsets hold");

        auto const o = getOffsets(object);
        Vec<TDim, Idx<T>> v;
        for(unsigned i = 0; i < TDim::value; i++)
            v[i] = o[(Dim<T>::value - TDim::value) + i];
        return v;
    }

    //! \return The offset in x dimension.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    AV2P_FN_HOST_ACC auto getOffsetX(TOffsets const& offsets = TOffsets()) -> Idx<TOffsets>
    {
        return getOffsets(offsets)[Dim<TOffsets>::value - 1u];
    }

    //! \return The offset in y dimension.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    AV2P_FN_HOST_ACC auto getOffsetY(TOffsets const& offsets = TOffsets()) -> Idx<TOffsets>
    {
        return getOffsets(offsets)[Dim<TOffsets>::value - 2u];
    }

    //! \return The offset in z dimension.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TOffsets>
    AV2P_FN_HOST_ACC auto getOffsetZ(TOffsets const& offsets = TOffsets()) -> Idx<TOffsets>
    {
        return getOffsets(offsets)[Dim<TOffsets>::value - 3u];
    }

    namespace trait
    {
        //! The Vec offset get trait specialization.
        template<typename TDim, typename TVal>
        struct GetOffsets<Vec<TDim, TVal>>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC constexpr auto operator()(Vec<TDim, TVal> const& offsets) const -> Vec<TDim, TVal>
            {
                return offsets;
            }
        };

        //! The unsigned integral x offset get trait specialization.
        template<typename TIntegral>
        struct GetOffsets<TIntegral, std::enable_if_t<std::is_integral_v<TIntegral>>>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC constexpr auto operator()(TIntegral const& i) const
            {
                return Vec{i};
            }
        };
    } // namespace trait
} // namespace aV2P
