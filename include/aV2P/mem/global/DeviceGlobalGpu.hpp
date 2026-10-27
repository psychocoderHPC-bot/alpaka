/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/BufTraits.hpp"
#include "aV2P/adapter/Queue.hpp"
#include "aV2P/adapter/ViewTraits.hpp"
#include "aV2P/mem/global/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"

#include <alpaka/alpaka.hpp>

#if ALPAKA_LANG_CUDA
#    include <alpaka/core/ApiCudaRt.hpp>
#endif
#if ALPAKA_LANG_HIP
#    include <alpaka/core/ApiHipRt.hpp>
#endif

#include <cstddef>
#include <type_traits>
#include <utility>

//! Device-global / static accelerator memory on the CUDA and HIP backends.
//!
//! 2.2 implemented this outside the generic CPU wrapper:
//!  - CUDA/HIP stored a `__device__`/`__constant__` symbol in
//!    `detail::DevGlobalImplGeneric<TagGpuCudaRt/TagGpuHipRt, T>` and resolved it with
//!    `ApiCudaRt/ApiHipRt::getSymbolAddress`
//!    (`alpaka/mem/global/DeviceGlobalUniformCudaHipBuiltIn.hpp`),
//!  - oneAPI used `sycl::ext::oneapi::experimental::device_global<T>`
//!    (`alpaka/mem/global/DeviceGlobalGenericSycl.hpp`).
//!
//! alpaka 3.0.0 provides the same feature through `ALPAKA_DEVICE_GLOBAL` /
//! `alpaka/onAcc/internal/globalMem.hpp`, but the adapter cannot reuse the macro: the legacy
//! aV2P macro is public 2.2 API.
//!
//! CUDA keeps the data in the legacy variable object because the adapter knows the concrete
//! accelerator type on the host, so the host and the kernel name the same variable-template
//! instantiation.  HIP re-instantiates the accelerator in the device pass (the wavefront size is
//! only known there), so the host and the kernel would otherwise address different variables.
//! HIP therefore uses one namespace-scope `__device__` storage object per (tag, element type).
//!
//! oneAPI is not specialized here.  There the kernel also receives a device-pass accelerator
//! instantiation, but a `sycl::device_global<T>` is tied to the variable object itself: a shared
//! namespace-scope `device_global` cannot be reached through the legacy `g<TAcc>` object, and a
//! static data member `device_global` is rejected/unresolved by DPC++ (verified with oneAPI
//! 2026.2 on the Arc A770).  The primary `DevGlobalTrait` therefore keeps its 2.2 diagnostic
//! instead of silently addressing the wrong symbol.  Removing that limitation needs either a
//! change to `AV2P_STATIC_ACC_MEM_GLOBAL` (aV2P core, outside this adapter patch) or a stable
//! public accelerator type across host and device (the oneAPI facade design).
namespace aV2P
{
    namespace detail
    {
#if(defined(AV2P_ACC_GPU_CUDA_ENABLED) && ALPAKA_LANG_CUDA) || (defined(AV2P_ACC_GPU_HIP_ENABLED) && ALPAKA_LANG_HIP)
        //! Remove const from the element type while keeping C array extents.
        template<typename T>
        struct RemoveConstElement
        {
            using type = std::remove_const_t<T>;
        };

        template<typename T, std::size_t N>
        struct RemoveConstElement<T[N]>
        {
            using type = typename RemoveConstElement<T>::type[N];
        };

#    if(defined(AV2P_ACC_GPU_HIP_ENABLED) && ALPAKA_LANG_HIP)
        //! Shared HIP device storage for one (tag, element type) pair.
        template<typename TTag, typename T>
        inline __device__ typename RemoveConstElement<T>::type deviceGlobalStorage;
#    endif

        //! Legacy `DevGlobal` value type for a CUDA/HIP backend tag.
        template<typename TTag, typename T>
        struct DevGlobalGpu
        {
            using Type = typename RemoveConstElement<T>::type;

#    if(defined(AV2P_ACC_GPU_CUDA_ENABLED) && ALPAKA_LANG_CUDA)
            // Host and kernel see the same public accelerator type, so the variable object can own
            // the device storage.  `AV2P_STATIC_ACC_MEM_GLOBAL` already places it in device memory.
            Type value;

            AV2P_FN_HOST_ACC T* operator&()
            {
                return &value;
            }

            AV2P_FN_HOST_ACC T& get()
            {
                return value;
            }
#    else
            // HIP: every TAcc instantiation of the legacy variable addresses one shared symbol.
            __device__ T* operator&()
            {
                return &deviceGlobalStorage<TTag, T>;
            }

            __device__ T& get()
            {
                return deviceGlobalStorage<TTag, T>;
            }
#    endif
        };

#    if(defined(AV2P_ACC_GPU_CUDA_ENABLED) && ALPAKA_LANG_CUDA)
        //! Resolve the legacy device-global symbol on CUDA.
        template<typename TTag, typename T>
        AV2P_FN_HOST auto deviceGlobalPtr(DevGlobalGpu<TTag, T>& global) -> std::remove_all_extents_t<T>*
        {
            static_assert(std::is_same_v<TTag, TagGpuCudaRt>);
            using ApiInterface = alpaka::ApiCudaRt;
            using TypeExt = typename RemoveConstElement<T>::type;
            TypeExt* pMemAcc = nullptr;
            // `&global` uses the wrapper's operator&, i.e. the address of the device variable.
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                ApiInterface,
                ApiInterface::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(const_cast<TypeExt*>(&global))));
            return reinterpret_cast<std::remove_all_extents_t<T>*>(pMemAcc);
        }
#    endif
#    if(defined(AV2P_ACC_GPU_HIP_ENABLED) && ALPAKA_LANG_HIP)
        //! Resolve the shared HIP device-global symbol.
        template<typename TTag, typename T>
        AV2P_FN_HOST auto deviceGlobalPtr(DevGlobalGpu<TTag, T>& /* global */) -> std::remove_all_extents_t<T>*
        {
            static_assert(std::is_same_v<TTag, TagGpuHipRt>);
            using ApiInterface = alpaka::ApiHipRt;
            using TypeExt = typename RemoveConstElement<T>::type;
            TypeExt* pMemAcc = nullptr;
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                ApiInterface,
                ApiInterface::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), deviceGlobalStorage<TTag, T>));
            return reinterpret_cast<std::remove_all_extents_t<T>*>(pMemAcc);
        }
#    endif
#endif
    } // namespace detail

    namespace detail
    {
#if(defined(AV2P_ACC_GPU_CUDA_ENABLED) && ALPAKA_LANG_CUDA)
        template<typename T>
        struct DevGlobalTrait<TagGpuCudaRt, T>
        {
            using Type = DevGlobalGpu<TagGpuCudaRt, T>;
        };
#endif
#if(defined(AV2P_ACC_GPU_HIP_ENABLED) && ALPAKA_LANG_HIP)
        template<typename T>
        struct DevGlobalTrait<TagGpuHipRt, T>
        {
            using Type = DevGlobalGpu<TagGpuHipRt, T>;
        };
#endif
    } // namespace detail

#if(defined(AV2P_ACC_GPU_CUDA_ENABLED) && ALPAKA_LANG_CUDA) || (defined(AV2P_ACC_GPU_HIP_ENABLED) && ALPAKA_LANG_HIP)
    //! Copy from host-accessible memory to a CUDA/HIP device global.
    template<
        concepts::Tag TTag,
        typename TQueue,
        typename TTypeDst,
        typename TViewSrc,
        typename std::enable_if_t<std::is_same_v<TTag, TagGpuCudaRt> || std::is_same_v<TTag, TagGpuHipRt>, int> = 0>
    AV2P_FN_HOST auto memcpy(TQueue& queue, detail::DevGlobalGpu<TTag, TTypeDst>& viewDst, TViewSrc const& viewSrc)
        -> void
    {
        using ElemType = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        auto src = adapter::makeAlpakaView(viewSrc);
        auto const extents = src.getExtents();
        auto const pitches = src.getPitches();
        // Writing a `__constant__` legacy global requires discarding element constness, exactly as
        // the 2.2 symbol copy did.
        auto dst = alpaka::makeView(
            queue.getAlpaka3().getDevice(),
            const_cast<ElemType*>(detail::deviceGlobalPtr(viewDst)),
            extents,
            pitches);
        alpaka::onHost::memcpy(queue.getAlpaka3(), dst, src, extents);
    }

    //! Copy from a CUDA/HIP device global to host-accessible memory.
    template<
        concepts::Tag TTag,
        typename TQueue,
        typename TViewDstFwd,
        typename TTypeSrc,
        typename std::enable_if_t<std::is_same_v<TTag, TagGpuCudaRt> || std::is_same_v<TTag, TagGpuHipRt>, int> = 0>
    AV2P_FN_HOST auto memcpy(TQueue& queue, TViewDstFwd&& viewDst, detail::DevGlobalGpu<TTag, TTypeSrc>& viewSrc)
        -> void
    {
        auto dst = adapter::makeAlpakaView(std::forward<TViewDstFwd>(viewDst));
        auto const extents = dst.getExtents();
        auto const pitches = dst.getPitches();
        auto src
            = alpaka::makeView(queue.getAlpaka3().getDevice(), detail::deviceGlobalPtr(viewSrc), extents, pitches);
        alpaka::onHost::memcpy(queue.getAlpaka3(), dst, src, extents);
    }

    //! Copy a sub-region from host-accessible memory to a CUDA/HIP device global.
    template<
        concepts::Tag TTag,
        typename TExtent,
        typename TQueue,
        typename TTypeDst,
        typename TViewSrc,
        typename std::enable_if_t<std::is_same_v<TTag, TagGpuCudaRt> || std::is_same_v<TTag, TagGpuHipRt>, int> = 0>
    AV2P_FN_HOST auto memcpy(
        TQueue& queue,
        detail::DevGlobalGpu<TTag, TTypeDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent const& extent) -> void
    {
        using ElemType = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        auto src = adapter::makeAlpakaView(viewSrc);
        auto dst = alpaka::makeView(
            queue.getAlpaka3().getDevice(),
            const_cast<ElemType*>(detail::deviceGlobalPtr(viewDst)),
            src.getExtents(),
            src.getPitches());
        alpaka::onHost::memcpy(queue.getAlpaka3(), dst, src, adapter::detail::toAlpakaExtent(extent));
    }

    //! Copy a sub-region from a CUDA/HIP device global to host-accessible memory.
    template<
        concepts::Tag TTag,
        typename TExtent,
        typename TQueue,
        typename TViewDstFwd,
        typename TTypeSrc,
        typename std::enable_if_t<std::is_same_v<TTag, TagGpuCudaRt> || std::is_same_v<TTag, TagGpuHipRt>, int> = 0>
    AV2P_FN_HOST auto memcpy(
        TQueue& queue,
        TViewDstFwd&& viewDst,
        detail::DevGlobalGpu<TTag, TTypeSrc>& viewSrc,
        TExtent const& extent) -> void
    {
        auto dst = adapter::makeAlpakaView(std::forward<TViewDstFwd>(viewDst));
        auto src = alpaka::makeView(
            queue.getAlpaka3().getDevice(),
            detail::deviceGlobalPtr(viewSrc),
            dst.getExtents(),
            dst.getPitches());
        alpaka::onHost::memcpy(queue.getAlpaka3(), dst, src, adapter::detail::toAlpakaExtent(extent));
    }
#endif
} // namespace aV2P
