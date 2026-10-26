/* Copyright 2026 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/adapter/Dev.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/platform/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>

namespace aV2P
{
    //! Platform handle wrapping an alpaka 3 `onHost::DeviceSelector`.
    //!
    //! `getAlpaka3()` returns a reference into the wrapper and is bound to the wrapper's lifetime.
    template<typename TDevSpec>
    class PlatformAlpaka3
        : public interface::Implements<ConceptPlatform, PlatformAlpaka3<TDevSpec>>
    {
    public:
        using DevSpec = TDevSpec;
        using DevType = DevAlpaka3<TDevSpec>;
        using Selector = decltype(alpaka::onHost::makeDeviceSelector(TDevSpec{}.toAlpaka()));

        PlatformAlpaka3() : m_selector{alpaka::onHost::makeDeviceSelector(TDevSpec{}.toAlpaka())}
        {
        }

        //! Non-owning reference to the wrapped alpaka 3 device selector.
        [[nodiscard]] auto getAlpaka3() & noexcept -> Selector&
        {
            return m_selector;
        }

        [[nodiscard]] auto getAlpaka3() const& noexcept -> Selector const&
        {
            return m_selector;
        }

    private:
        mutable Selector m_selector;
    };

    using PlatformCpu = PlatformAlpaka3<adapter::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>>;

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    using PlatformCudaRt = PlatformAlpaka3<adapter::DeviceSpec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>>;
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    using PlatformHipRt = PlatformAlpaka3<adapter::DeviceSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>>;
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    using PlatformCpuSycl = PlatformAlpaka3<adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::Cpu>>;
    using PlatformGpuSyclIntel
        = PlatformAlpaka3<adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::IntelGpu>>;
#endif

    namespace adapter
    {
#ifdef AV2P_ACC_SYCL_ENABLED
        //! Platform type behind `PlatformGenericSycl<TTag>`.
        template<typename TTag, typename = void>
        struct GenericSyclPlatformType
        {
            using type = legacy::UnimplementedBackend<TTag>;
        };

        template<typename TTag>
        struct GenericSyclPlatformType<TTag, std::void_t<typename LegacyAccMapping<TTag>::Api>>
        {
            using type = PlatformAlpaka3<DeviceSpec<
                typename LegacyAccMapping<TTag>::Api,
                typename LegacyAccMapping<TTag>::DeviceKind>>;
        };
#endif

        //! Platform type behind `PlatformUniformCudaHipRt<TApi>`.
        template<typename TApi, typename = void>
        struct UniformCudaHipPlatformType
        {
            using type = legacy::UnimplementedBackend<TApi>;
        };

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
        template<>
        struct UniformCudaHipPlatformType<ApiCudaRt, void>
        {
            using type = PlatformCudaRt;
        };
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
        template<>
        struct UniformCudaHipPlatformType<ApiHipRt, void>
        {
            using type = PlatformHipRt;
        };
#endif
    } // namespace adapter

#ifdef AV2P_ACC_SYCL_ENABLED
    //! 2.2 platform name: the platform of the oneAPI tag `TTag`.
    template<typename TTag>
    using PlatformGenericSycl = typename adapter::GenericSyclPlatformType<TTag>::type;

    using PlatformGpuSyclAmd = PlatformGenericSycl<TagGpuSyclAmd>;
    using PlatformGpuSyclNvidia = PlatformGenericSycl<TagGpuSyclNvidia>;
#endif

    //! 2.2 platform name: the CUDA or HIP platform selected by the legacy runtime API tag.
    template<typename TApi>
    using PlatformUniformCudaHipRt = typename adapter::UniformCudaHipPlatformType<TApi>::type;

    namespace trait
    {
        template<typename TDevSpec>
        struct DevType<PlatformAlpaka3<TDevSpec>>
        {
            using type = DevAlpaka3<TDevSpec>;
        };

        template<typename TDevSpec>
        struct GetDevCount<PlatformAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto getDevCount(PlatformAlpaka3<TDevSpec> const& platform) -> std::size_t
            {
                return platform.getAlpaka3().getDeviceCount();
            }
        };

        template<typename TDevSpec>
        struct GetDevByIdx<PlatformAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto getDevByIdx(PlatformAlpaka3<TDevSpec> const& platform, std::size_t devIdx)
                -> DevAlpaka3<TDevSpec>
            {
                // `DeviceSelector::makeDevice()` is non-const in alpaka 3; the selector itself is
                // stored in the platform wrapper, so no selector is recreated here.
                auto& selector = const_cast<typename PlatformAlpaka3<TDevSpec>::Selector&>(platform.getAlpaka3());
                return DevAlpaka3<TDevSpec>{selector.makeDevice(static_cast<std::uint32_t>(devIdx))};
            }
        };
    } // namespace trait
} // namespace aV2P
