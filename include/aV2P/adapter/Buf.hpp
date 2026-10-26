/* Copyright 2026 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/adapter/Dev.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>
#include <utility>

namespace aV2P::detail
{
    //! Keep the allocation's alignment only when the whole buffer is one row.
    //!
    //! A 2.2 contiguous re-view of a multi-dimensional allocation drops the padding that made every
    //! interior row start SIMD aligned.  Advertising the allocation alignment on such a view lets
    //! native alpaka 3 operations issue misaligned SIMD accesses.  The element alignment is the
    //! strongest guarantee the contiguous layout actually provides for more than one dimension.
    template<typename TBuffer, bool TKeepAllocationAlignment>
    struct NativeBufferWithMatchingAlignment
    {
        using type = TBuffer;
    };

    template<typename TApi, typename TType, typename TExtents, typename TAlignment>
    struct NativeBufferWithMatchingAlignment<
        alpaka::onHost::SharedBuffer<TApi, TType, TExtents, TAlignment>,
        false>
    {
        using type = alpaka::onHost::SharedBuffer<TApi, TType, TExtents, alpaka::Alignment<>>;
    };

    /*! Pick the buffer type a buffer wrapper stores.
     *
     * Host-API allocations are re-viewed with the 2.2 contiguous pitches, so a multi-dimensional
     * buffer only promises element alignment (see `NativeBufferWithMatchingAlignment`).  Device-API
     * allocations keep the native alpaka 3 type, which carries the native padded pitches and the
     * native alignment of the allocation.  `TIsHostApi` is the API tag check and
     * `TIsOneDimensional` is `TDim::value <= 1u`.
     */
    template<typename TDevSpec, typename TBuffer, bool TIsHostApi, bool TIsOneDimensional>
    struct NativeBufferForSpec
    {
        using type = TBuffer;
    };

    template<typename TDevSpec, typename TBuffer, bool TIsOneDimensional>
    struct NativeBufferForSpec<TDevSpec, TBuffer, true, TIsOneDimensional>
    {
        using type = typename NativeBufferWithMatchingAlignment<TBuffer, TIsOneDimensional>::type;
    };
} // namespace aV2P::detail

namespace aV2P
{
    template<typename TDevSpec, typename TElem, typename TDim, typename TIdx>
    class ConstBufAlpaka3;

    //! Memory buffer wrapping an alpaka 3 `SharedBuffer`.
    //!
    //! `getAlpaka3()` returns a reference into the wrapper; it is non-owning and bound to the
    //! wrapper's lifetime.
    template<typename TDevSpec, typename TElem, typename TDim, typename TIdx>
    class BufAlpaka3 : public internal::ViewAccessorType<DevAlpaka3<TDevSpec>, BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
    public:
        friend ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>;

        using DevType = DevAlpaka3<TDevSpec>;
        using AllocatedBufferType = decltype(alpaka::onHost::alloc<TElem>(
            std::declval<DevType&>().getAlpaka3(),
            std::declval<alpaka::Vec<TIdx, static_cast<std::uint32_t>(TDim::value)>>()));
        using AlpakaBufferType = typename detail::NativeBufferForSpec<
            TDevSpec,
            AllocatedBufferType,
            std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>,
            (TDim::value <= 1u)>::type;

        [[nodiscard]] auto getAlpaka3() & noexcept -> AlpakaBufferType&
        {
            return m_memImpl;
        }

        [[nodiscard]] auto getAlpaka3() const& noexcept -> AlpakaBufferType const&
        {
            return m_memImpl;
        }

        template<typename TExtent, typename TPitch, typename Deleter>
        AV2P_FN_HOST BufAlpaka3(
            DevAlpaka3<TDevSpec> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extents,
            TPitch const& pitches)
            : m_memImpl{
                  AlpakaBufferType(alpaka::getApi(dev.getAlpaka3()), pMem, extents, pitches, std::move(deleter))}
            , m_dev{dev}
        {
        }

        AV2P_FN_HOST BufAlpaka3(AlpakaBufferType const& sharedBuffer, DevAlpaka3<TDevSpec> const& dev)
            : m_memImpl{sharedBuffer}
            , m_dev{dev}
        {
        }

        BufAlpaka3(BufAlpaka3 const&) = default;
        auto operator=(BufAlpaka3 const&) -> BufAlpaka3& = default;
        BufAlpaka3(BufAlpaka3&&) = default;
        auto operator=(BufAlpaka3&&) -> BufAlpaka3& = default;

        [[nodiscard]] AV2P_FN_HOST auto getApi() const
        {
            return alpaka::getApi(m_memImpl);
        }

        [[nodiscard]] AV2P_FN_HOST auto getExtents() const
        {
            return m_memImpl.getExtents();
        }

        [[nodiscard]] AV2P_FN_HOST auto getPitches() const
        {
            return m_memImpl.getPitches();
        }

        [[nodiscard]] AV2P_FN_HOST auto data() const
        {
            return m_memImpl.data();
        }

        [[nodiscard]] AV2P_FN_HOST auto data()
        {
            return m_memImpl.data();
        }

    public:
        AlpakaBufferType m_memImpl;
        DevAlpaka3<TDevSpec> m_dev;
    };

    template<typename TDevSpec, typename TElem, typename TDim, typename TIdx>
    class ConstBufAlpaka3
        : public internal::ViewAccessorType<
              DevAlpaka3<TDevSpec>,
              ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
    public:
        using DevType = DevAlpaka3<TDevSpec>;
        using AllocatedBufferType = decltype(alpaka::onHost::alloc<TElem>(
            std::declval<DevType&>().getAlpaka3(),
            std::declval<alpaka::Vec<TIdx, static_cast<std::uint32_t>(TDim::value)>>()));
        using AlpakaBufferType = typename detail::NativeBufferForSpec<
            TDevSpec,
            AllocatedBufferType,
            std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>,
            (TDim::value <= 1u)>::type;

        [[nodiscard]] auto getAlpaka3() & noexcept -> AlpakaBufferType&
        {
            return m_memImpl;
        }

        [[nodiscard]] auto getAlpaka3() const& noexcept -> AlpakaBufferType const&
        {
            return m_memImpl;
        }

        template<typename TExtent, typename TPitch, typename Deleter>
        AV2P_FN_HOST ConstBufAlpaka3(
            DevAlpaka3<TDevSpec> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extents,
            TPitch const& pitches)
            : m_memImpl{
                  AlpakaBufferType(alpaka::getApi(dev.getAlpaka3()), pMem, extents, pitches, std::move(deleter))}
            , m_dev{dev}
        {
        }

        AV2P_FN_HOST ConstBufAlpaka3(AlpakaBufferType const& sharedBuffer, DevAlpaka3<TDevSpec> const& dev)
            : m_memImpl{sharedBuffer}
            , m_dev{dev}
        {
        }

        AV2P_FN_HOST ConstBufAlpaka3(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            : m_memImpl{buf.m_memImpl}
            , m_dev{buf.m_dev}
        {
        }

        AV2P_FN_HOST ConstBufAlpaka3(BufAlpaka3<TDevSpec, TElem, TDim, TIdx>&& buf)
            : m_memImpl{std::move(buf.m_memImpl)}
            , m_dev{std::move(buf.m_dev)}
        {
        }

        [[nodiscard]] AV2P_FN_HOST auto getApi() const
        {
            return alpaka::getApi(m_memImpl);
        }

        [[nodiscard]] AV2P_FN_HOST auto getExtents() const
        {
            return m_memImpl.getExtents();
        }

        [[nodiscard]] AV2P_FN_HOST auto getPitches() const
        {
            return m_memImpl.getPitches();
        }

        [[nodiscard]] AV2P_FN_HOST auto data() const
        {
            return m_memImpl.data();
        }

    public:
        AlpakaBufferType m_memImpl;
        DevAlpaka3<TDevSpec> m_dev;
    };

    template<typename TElem, typename TDim, typename TIdx>
    using BufCpu = BufAlpaka3<
        adapter::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
        TElem,
        TDim,
        TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufCpu = ConstBufAlpaka3<
        adapter::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
        TElem,
        TDim,
        TIdx>;

    //! 2.2 buffer names lost when the per-backend buffer implementations were removed.
    //!
    //! The element/dimension/index arity matches 2.2.  Backends with an alpaka 3 executor map to the
    //! adapter buffer; the oneAPI device kinds without an executor keep a name-only placeholder.
#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    template<typename TElem, typename TDim, typename TIdx>
    using BufCudaRt = BufAlpaka3<
        adapter::DeviceSpec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        TElem,
        TDim,
        TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufCudaRt = ConstBufAlpaka3<
        adapter::DeviceSpec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        TElem,
        TDim,
        TIdx>;
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    template<typename TElem, typename TDim, typename TIdx>
    using BufHipRt = BufAlpaka3<
        adapter::DeviceSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        TElem,
        TDim,
        TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufHipRt = ConstBufAlpaka3<
        adapter::DeviceSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        TElem,
        TDim,
        TIdx>;
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
#    if defined(AV2P_SYCL_TARGET_CPU)
    template<typename TElem, typename TDim, typename TIdx>
    using BufCpuSycl = BufAlpaka3<
        adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::Cpu>,
        TElem,
        TDim,
        TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufCpuSycl = ConstBufAlpaka3<
        adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::Cpu>,
        TElem,
        TDim,
        TIdx>;
#    endif

#    if defined(AV2P_SYCL_TARGET_GPU)
    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclIntel = BufAlpaka3<
        adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::IntelGpu>,
        TElem,
        TDim,
        TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufGpuSyclIntel = ConstBufAlpaka3<
        adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::IntelGpu>,
        TElem,
        TDim,
        TIdx>;
#    endif

    //! Name-only buffer placeholder for oneAPI device kinds without an alpaka 3 executor.
    namespace legacy_buf
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct UnimplementedBuf : legacy::UnimplementedBackend<TElem>
        {
        };
    } // namespace legacy_buf

    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclAmd = legacy_buf::UnimplementedBuf<TElem, TDim, TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufGpuSyclAmd = legacy_buf::UnimplementedBuf<TElem, TDim, TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclNvidia = legacy_buf::UnimplementedBuf<TElem, TDim, TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufGpuSyclNvidia = legacy_buf::UnimplementedBuf<TElem, TDim, TIdx>;
#endif
} // namespace aV2P
