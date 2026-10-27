/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/math/Complex.hpp"
#include "aV2P/math/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace aV2P::math
{
    namespace trait
    {
        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Abs<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_signed_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::abs(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Acos<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::acos(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Acosh<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::acosh(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Asin<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::asin(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Asinh<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::asinh(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Atan<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::atan(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Atanh<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::atanh(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Cbrt<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::cbrt(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Ceil<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::ceil(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Conj<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::conj(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Cos<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::cos(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Cosh<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::cosh(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Erf<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::erf(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Exp<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::exp(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Floor<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::floor(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Isfinite<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::isfinite(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Isinf<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::isinf(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Isnan<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::isnan(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Log<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::log(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Log2<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::log2(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Log10<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::log10(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Round<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::round(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Lround<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::lround(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Llround<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::llround(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Rsqrt<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::rsqrt(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Sin<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::sin(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Sinh<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::sinh(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Sqrt<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::sqrt(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Tan<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::tan(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Tanh<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::tanh(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Trunc<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg) const
            {
                return alpaka::math::trunc(arg);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Arg<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& argument) const
            {
                return alpaka::math::arg(argument);
            }
        };

        template<
            typename Ty,
            typename Tx,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Atan2<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            Ty,
            Tx,
            std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                Ty const& y,
                Tx const& x) const
            {
                return alpaka::math::atan2(y, x);
            }
        };

        template<
            typename TMag,
            typename TSgn,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Copysign<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TMag,
            TSgn,
            std::enable_if_t<std::is_floating_point_v<TMag> && std::is_floating_point_v<TSgn>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TMag const& mag,
                TSgn const& sgn) const
            {
                return alpaka::math::copysign(mag, sgn);
            }
        };

        //! Scalar forwarding to the alpaka 3 math API.
        //!
        //! Restrict the adapter to arithmetic arguments like the 2.2 CUDA/HIP and standard-library
        //! `Min` specializations do.  Claiming every non-complex type would make this partial
        //! specialization compete with client specializations that supply their own vector or
        //! aggregate `Min` (for example the `pmacc::math::Vector` one), yielding an ambiguous
        //! partial-specialization set for `Min<AccAlpaka3<...>, T, T, void>`.
        template<
            typename Tx,
            typename Ty,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Min<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            Tx,
            Ty,
            std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                Tx const& x,
                Ty const& y) const
            {
                return alpaka::math::min(x, y);
            }
        };

        //! Scalar forwarding to the alpaka 3 math API.
        //!
        //! Restrict the adapter to arithmetic arguments like the 2.2 CUDA/HIP and standard-library
        //! `Max` specializations do.  Claiming every non-complex type would make this partial
        //! specialization compete with client specializations that supply their own vector or
        //! aggregate `Max` (for example the `pmacc::math::Vector` one), yielding an ambiguous
        //! partial-specialization set for `Max<AccAlpaka3<...>, T, T, void>`.
        template<
            typename Tx,
            typename Ty,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Max<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            Tx,
            Ty,
            std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                Tx const& x,
                Ty const& y) const
            {
                return alpaka::math::max(x, y);
            }
        };

        template<
            typename Tx,
            typename Ty,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Fmod<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            Tx,
            Ty,
            std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                Tx const& x,
                Ty const& y) const
            {
                return alpaka::math::fmod(x, y);
            }
        };

        template<
            typename Tx,
            typename Ty,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Remainder<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            Tx,
            Ty,
            std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                Tx const& x,
                Ty const& y) const
            {
                return alpaka::math::remainder(x, y);
            }
        };

        template<
            typename TBase,
            typename TExp,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Pow<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TBase,
            TExp,
            std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TBase const& base,
                TExp const& exp) const
            {
                return alpaka::math::pow(base, exp);
            }
        };

        template<
            typename Tx,
            typename Ty,
            typename Tz,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Fma<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            Tx,
            Ty,
            Tz,
            std::enable_if_t<
                std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty> && std::is_floating_point_v<Tz>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                Tx const& x,
                Ty const& y,
                Tz const& z) const
            {
                return alpaka::math::fma(x, y, z);
            }
        };

        template<
            typename TArg,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct SinCos<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TArg,
            std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            AV2P_FN_HOST_ACC auto operator()(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                TArg const& arg,
                TArg& result_sin,
                TArg& result_cos) const -> void
            {
                alpaka::math::sincos(arg, result_sin, result_cos);
            }
        };

    } // namespace trait
} // namespace aV2P::math
