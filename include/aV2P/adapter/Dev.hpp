/* Copyright 2026 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber,
 *                Antonio Di Pilato, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/platform/Traits.hpp"
#include "aV2P/queue/Properties.hpp"
#include "aV2P/queue/Traits.hpp"
#include "aV2P/traits/Traits.hpp"
#include "aV2P/wait/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace aV2P
{
    template<typename TDevSpec>
    class DevAlpaka3;

    template<typename TDevSpec>
    class PlatformAlpaka3;

    template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
    class QueueAlpaka3;

    namespace trait
    {
        template<typename TPlatform, typename TSfinae>
        struct GetDevByIdx;
    } // namespace trait

    //! Device handle wrapping an alpaka 3 `onHost::Device`.
    //!
    //! The wrapped alpaka 3 device is stored by value (alpaka 3 handles are cheap, shared-ownership
    //! values).  `getAlpaka3()` returns a reference into the wrapper; it is non-owning and must not
    //! outlive the `DevAlpaka3` instance.
    template<typename TDevSpec>
    class DevAlpaka3
        : public interface::Implements<ConceptCurrentThreadWaitFor, DevAlpaka3<TDevSpec>>
        , public interface::Implements<ConceptDev, DevAlpaka3<TDevSpec>>
    {
        friend struct trait::GetDevByIdx<PlatformAlpaka3<TDevSpec>>;

    protected:
        explicit DevAlpaka3(std::size_t devIdx)
            : m_devImpl{
                  alpaka::onHost::makeDeviceSelector(DevSpec{}.toAlpaka()).makeDevice(static_cast<std::uint32_t>(devIdx))}
        {
        }

    public:
        using DevSpec = TDevSpec;
        using DevType = decltype(alpaka::onHost::makeDeviceSelector(DevSpec{}.toAlpaka()).makeDevice(0));

        AV2P_FN_HOST explicit DevAlpaka3(DevType const& alpaka3Dev) : m_devImpl{alpaka3Dev}
        {
        }

        //! Non-owning reference to the wrapped alpaka 3 device.
        //! The reference is valid only while this wrapper is alive.
        [[nodiscard]] auto getAlpaka3() & noexcept -> DevType&
        {
            return m_devImpl;
        }

        //! Non-owning const reference to the wrapped alpaka 3 device.
        [[nodiscard]] auto getAlpaka3() const& noexcept -> DevType const&
        {
            return m_devImpl;
        }

        auto operator==(DevAlpaka3 const& other) const -> bool
        {
            return m_devImpl == other.m_devImpl;
        }

        auto operator!=(DevAlpaka3 const& other) const -> bool
        {
            return m_devImpl != other.m_devImpl;
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            using NativeHandleType = decltype(m_devImpl.getNativeHandle());
            if constexpr(std::is_integral_v<NativeHandleType>)
            {
                // alpaka 2.2 exposed every integral device handle as `int` (see the 2.2
                // DevCpu/DevUniformCudaHipRt implementations).  alpaka 3 uses `uint32_t`
                // for the host backend and `int` for CUDA/HIP, so restore the legacy type
                // at this boundary.  Non-integral handles (SYCL device/context pair) pass
                // through unchanged, matching their 2.2 form.
                return static_cast<int>(m_devImpl.getNativeHandle());
            }
            else
            {
                return m_devImpl.getNativeHandle();
            }
        }

        friend struct trait::GetName<DevAlpaka3>;
        friend struct trait::GetMemBytes<DevAlpaka3>;
        friend struct trait::GetFreeMemBytes<DevAlpaka3>;
        friend struct trait::GetWarpSizes<DevAlpaka3>;
        friend struct trait::GetPreferredWarpSize<DevAlpaka3>;

    private:
        DevType m_devImpl;
    };

    using DevCpu = DevAlpaka3<adapter::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>>;

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    using DevCudaRt = DevAlpaka3<adapter::DeviceSpec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>>;
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    using DevHipRt = DevAlpaka3<adapter::DeviceSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>>;
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    using DevCpuSycl = DevAlpaka3<adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::Cpu>>;
    using DevGpuSyclIntel = DevAlpaka3<adapter::DeviceSpec<alpaka::api::OneApi, alpaka::deviceKind::IntelGpu>>;
#endif

    namespace trait
    {
        template<typename TDevSpec>
        struct GetName<DevAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto getName(DevAlpaka3<TDevSpec> const& dev) -> std::string
            {
                return dev.getAlpaka3().getName();
            }
        };

        template<typename TDevSpec>
        struct GetMemBytes<DevAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto getMemBytes(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3().getDeviceProperties().globalMemCapacityBytes;
            }
        };

        template<typename TDevSpec>
        struct GetFreeMemBytes<DevAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto getFreeMemBytes(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3().getFreeGlobalMemBytes();
            }
        };

        template<typename TDevSpec>
        struct GetWarpSizes<DevAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto getWarpSizes(DevAlpaka3<TDevSpec> const& dev) -> std::vector<std::size_t>
            {
                return {dev.getAlpaka3().getDeviceProperties().warpSize};
            }
        };

        template<typename TDevSpec>
        struct GetPreferredWarpSize<DevAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto getPreferredWarpSize(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3().getDeviceProperties().warpSize;
            }
        };

        //! alpaka 3 has no device-reset operation.  Keep the legacy diagnostic instead of silently
        //! doing something different.
        template<typename TDevSpec>
        struct Reset<DevAlpaka3<TDevSpec>>
        {
            AV2P_FN_HOST static auto reset(DevAlpaka3<TDevSpec> const& /* dev */) -> void
            {
                static_assert(
                    sizeof(TDevSpec) == 0,
                    "aV2P Phase 3a: reset(device) has no alpaka 3 equivalent and is not implemented.");
            }
        };

        template<typename TDevSpec>
        struct NativeHandle<DevAlpaka3<TDevSpec>>
        {
            [[nodiscard]] static auto getNativeHandle(DevAlpaka3<TDevSpec> const& dev)
            {
                return dev.getNativeHandle();
            }
        };

        template<typename TDevSpec>
        struct PlatformType<DevAlpaka3<TDevSpec>>
        {
            using type = PlatformAlpaka3<TDevSpec>;
        };

        template<typename TDevSpec>
        struct QueueType<DevAlpaka3<TDevSpec>, Blocking>
        {
            using type = QueueAlpaka3<DevAlpaka3<TDevSpec>, alpaka::queueKind::Blocking>;
        };

        template<typename TDevSpec>
        struct QueueType<DevAlpaka3<TDevSpec>, NonBlocking>
        {
            using type = QueueAlpaka3<DevAlpaka3<TDevSpec>, alpaka::queueKind::NonBlocking>;
        };
    } // namespace trait
} // namespace aV2P
