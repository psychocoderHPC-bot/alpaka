/* Copyright 2026 Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/dev/DevGenericSycl.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"

#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace aV2P
{
} // namespace aV2P

namespace aV2P::concepts
{
    template<typename TView>
    concept View = requires {
        typename Idx<TView>;
        typename Dim<TView>;
        {
            aV2P::getPtrNative(std::declval<TView>())
        };
        {
            aV2P::getPitchesInBytes(std::declval<TView>())
        };
        {
            aV2P::getExtents(std::declval<TView>())
        };
    };
} // namespace aV2P::concepts

namespace aV2P::internal
{

    template<concepts::View TView>
    struct BaseViewAccessor
    {
    private:
        using value_type = Elem<TView>;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;
        using Idx = aV2P::Idx<TView>;
        using Dim = aV2P::Dim<TView>;

    public:
        [[nodiscard]] AV2P_FN_HOST auto data() -> pointer
        {
            return aV2P::getPtrNative(*static_cast<TView*>(this));
        }

        [[nodiscard]] AV2P_FN_HOST auto data() const -> const_pointer
        {
            return aV2P::getPtrNative(*static_cast<TView const*>(this));
        }

        AV2P_FN_HOST auto begin() -> pointer requires(Dim::value == 1)
        {
            return data();
        }

        AV2P_FN_HOST auto begin() const -> const_pointer requires(Dim::value == 1)
        {
            return data();
        }

        AV2P_FN_HOST auto cbegin() const -> const_pointer requires(Dim::value == 1)
        {
            return data();
        }

        AV2P_FN_HOST auto end() -> pointer requires(Dim::value == 1)
        {
            return data() + aV2P::getExtents(*static_cast<TView*>(this))[0];
        }

        AV2P_FN_HOST auto end() const -> const_pointer requires(Dim::value == 1)
        {
            return data() + aV2P::getExtents(*static_cast<TView const*>(this))[0];
        }

        AV2P_FN_HOST auto cend() const -> const_pointer requires(Dim::value == 1)
        {
            return data() + aV2P::getExtents(*static_cast<TView const*>(this))[0];
        }

        AV2P_FN_HOST auto rank() const -> Idx
        {
            return Dim::value;
        }

        AV2P_FN_HOST auto size() const -> Idx requires(Dim::value == 1)
        {
            return aV2P::getExtents(*static_cast<TView const*>(this))[0];
        }

        AV2P_FN_HOST auto size() const -> Idx requires(Dim::value > 1)
        {
            return aV2P::getExtents(*static_cast<TView const*>(this)).prod();
        }

        AV2P_FN_HOST auto extent(Idx dim) const -> Idx
        {
            return aV2P::getExtents(*static_cast<TView const*>(this))[dim];
        }

        AV2P_FN_HOST auto extents() const -> Vec<Dim, Idx>;

#if AV2P_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif
        AV2P_FN_HOST operator std::span<value_type const>() const requires(Dim::value == 1)
        {
            return std::span<value_type const>{begin(), end()};
        }

        AV2P_FN_HOST operator std::span<value_type>() requires(Dim::value == 1)
        {
            return std::span<value_type>{begin(), end()};
        }
#if AV2P_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    };

    template<concepts::View TView>
    using DeviceViewAccessor = BaseViewAccessor<TView>;

    template<concepts::View TView>
    struct HostViewAccessor : BaseViewAccessor<TView>
    {
    private:
        using value_type = Elem<TView>;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;
        using Idx = aV2P::Idx<TView>;
        using Dim = aV2P::Dim<TView>;

    public:
        AV2P_FN_HOST auto operator*() -> reference
        {
            static_assert(Dim::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *(this->data());
        }

        AV2P_FN_HOST auto operator*() const -> const_reference
        {
            static_assert(Dim::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *(this->data());
        }

        AV2P_FN_HOST auto operator->() -> pointer
        {
            static_assert(Dim::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return this->data();
        }

        AV2P_FN_HOST auto operator->() const -> const_pointer
        {
            static_assert(Dim::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return this->data();
        }

        AV2P_FN_HOST auto operator[](Idx i) -> reference
        {
            static_assert(Dim::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return this->data()[i];
        }

        AV2P_FN_HOST auto operator[](Idx i) const -> const_reference
        {
            static_assert(Dim::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return this->data()[i];
        }

    private:
        template<typename TIdx>
        [[nodiscard]] AV2P_FN_HOST auto ptr_at([[maybe_unused]] Vec<Dim, TIdx> index) const -> const_pointer
        {
            static_assert(
                std::is_convertible_v<TIdx, Idx>,
                "the index type must be convertible to the index of the Buffer or View");

            auto ptr = reinterpret_cast<std::uintptr_t>(this->data());
            if constexpr(Dim::value > 0)
            {
                ptr += static_cast<std::uintptr_t>(
                    (aV2P::getPitchesInBytes(*static_cast<TView const*>(this)) * aV2P::castVec<Idx>(index)).sum());
            }
            return reinterpret_cast<const_pointer>(ptr);
        }

    public:
        template<typename TIdx>
        AV2P_FN_HOST auto operator[](Vec<Dim, TIdx> index) -> reference
        {
            return *const_cast<pointer>(ptr_at(index));
        }

        template<typename TIdx>
        AV2P_FN_HOST auto operator[](Vec<Dim, TIdx> index) const -> const_reference
        {
            return *ptr_at(index);
        }

        template<typename TIdx>
        AV2P_FN_HOST auto at(Vec<Dim, TIdx> index) -> reference
        {
            auto extent = aV2P::getExtents(*static_cast<TView*>(this));
            if(!(index < extent).all())
            {
                std::stringstream msg;
                msg << "index " << index << " is outside of the Buffer or View extent " << extent;
                throw std::out_of_range(msg.str());
            }
            return *const_cast<pointer>(ptr_at(index));
        }

        template<typename TIdx>
        [[nodiscard]] AV2P_FN_HOST auto at(Vec<Dim, TIdx> index) const -> const_reference
        {
            auto extent = aV2P::getExtents(*static_cast<TView const*>(this));
            if(!(index < extent).all())
            {
                std::stringstream msg;
                msg << "index " << index << " is outside of the Buffer or View extent " << extent;
                throw std::out_of_range(msg.str());
            }
            return *ptr_at(index);
        }
    };

    template<typename TDev>
    struct ViewAccessor
    {
        template<concepts::View TView>
        using AccessorType = DeviceViewAccessor<TView>;
    };

    template<>
    struct ViewAccessor<aV2P::DevCpu>
    {
        template<concepts::View TView>
        using AccessorType = HostViewAccessor<TView>;
    };

#ifdef AV2P_ACC_SYCL_ENABLED
    template<>
    struct ViewAccessor<aV2P::DevGenericSycl<aV2P::TagCpuSycl>>
    {
        template<concepts::View TView>
        using AccessorType = HostViewAccessor<TView>;
    };
#endif

    template<typename TDev, concepts::View TView>
    using ViewAccessorType = typename ViewAccessor<TDev>::template AccessorType<TView>;

} // namespace aV2P::internal
