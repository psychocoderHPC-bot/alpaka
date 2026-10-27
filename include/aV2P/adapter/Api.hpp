/* Copyright 2026 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Andrea Bocci,
 *                Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Legacy CUDA/HIP runtime API tags and the runtime error-checking macros.
//!
//! The 2.2 runtime wrappers (`ApiCudaRt`, `ApiHipRt`) are gone: alpaka 3 issues every runtime call
//! internally.  The public names survive as empty tag types so that legacy code can keep using them as
//! template arguments (for example `AccGpuUniformCudaHipRt<TApi, TDim, TIdx>`).  A pure host
//! translation unit that only names the tag does not need a vendor header.
//!
//! The `AV2P_UNIFORM_CUDA_HIP_RT_CHECK*` macros are kept because callers (the CUDA/HIP test helpers)
//! still spell them.  They only exist in a CUDA/HIP build.

#include "aV2P/core/Common.hpp" // AV2P_FN_HOST, AV2P_LANG_*, AV2P_DEBUG, AV2P_DEBUG_BREAK

#if AV2P_LANG_CUDA || defined(AV2P_ACC_GPU_CUDA_ENABLED)
#    include <cuda_runtime.h>
#endif
#if AV2P_LANG_HIP || defined(AV2P_ACC_GPU_HIP_ENABLED)
#    include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

namespace aV2P
{
    //! Legacy CUDA runtime API tag; the 2.2 wrapper body is replaced by alpaka 3.
    struct ApiCudaRt
    {
#ifdef AV2P_ACC_GPU_CUDA_ENABLED
        using Error_t = cudaError_t;
        using Stream_t = cudaStream_t;

        static constexpr Error_t success = cudaSuccess;

        static auto getLastError() -> Error_t
        {
            return cudaGetLastError();
        }

        static auto getErrorName(Error_t error) -> char const*
        {
            return cudaGetErrorName(error);
        }

        static auto getErrorString(Error_t error) -> char const*
        {
            return cudaGetErrorString(error);
        }

        static auto streamCreate(Stream_t* stream) -> Error_t
        {
            return cudaStreamCreate(stream);
        }

        static auto streamDestroy(Stream_t stream) -> Error_t
        {
            return cudaStreamDestroy(stream);
        }
#endif
    };

    //! Legacy HIP runtime API tag; the 2.2 wrapper body is replaced by alpaka 3.
    struct ApiHipRt
    {
#ifdef AV2P_ACC_GPU_HIP_ENABLED
        using Error_t = hipError_t;
        using Stream_t = hipStream_t;

        static constexpr Error_t success = hipSuccess;

        static auto getLastError() -> Error_t
        {
            return hipGetLastError();
        }

        static auto getErrorName(Error_t error) -> char const*
        {
            return hipGetErrorName(error);
        }

        static auto getErrorString(Error_t error) -> char const*
        {
            return hipGetErrorString(error);
        }

        static auto streamCreate(Stream_t* stream) -> Error_t
        {
            return hipStreamCreate(stream);
        }

        static auto streamDestroy(Stream_t stream) -> Error_t
        {
            return hipStreamDestroy(stream);
        }
#endif
    };
} // namespace aV2P

// Language guards moved from the deleted `core/Hip.hpp` (the CUDA equivalent lived in `core/Cuda.hpp`).
#if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA && !defined(AV2P_HOST_ONLY)
#    error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP && !defined(AV2P_HOST_ONLY)
#    error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P::uniform_cuda_hip::detail
{
    //! CUDA/HIP runtime API error checking with log and exception, ignoring specific error values
    template<typename TApi, bool TThrow>
    AV2P_FN_HOST inline void rtCheck(
        typename TApi::Error_t const& error,
        char const* desc,
        char const* file,
        int const& line) noexcept(!TThrow)
    {
        if(error != TApi::success)
        {
            auto const sError = std::string{
                std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '"
                + TApi::getErrorName(error) + "': '" + std::string(TApi::getErrorString(error)) + "'!"};

            if constexpr(!TThrow || AV2P_DEBUG >= AV2P_DEBUG_MINIMAL)
                std::cerr << sError << std::endl;

            AV2P_DEBUG_BREAK;
            // reset the last error to allow user side error handling. Using std::ignore to discard unneeded
            // return values is suggested by the C++ core guidelines.
            std::ignore = TApi::getLastError();

            if constexpr(TThrow)
                throw std::runtime_error(sError);
        }
    }

    //! CUDA/HIP runtime API error checking with log and exception, ignoring specific error values
    template<typename TApi, bool TThrow>
    AV2P_FN_HOST inline void rtCheckIgnore(
        typename TApi::Error_t const& error,
        char const* cmd,
        char const* file,
        int const& line,
        std::initializer_list<typename TApi::Error_t> ignoredErrorCodes) noexcept(!TThrow)
    {
        if(error != TApi::success)
        {
            // If the error code is not one of the ignored ones.
            if(std::find(std::cbegin(ignoredErrorCodes), std::cend(ignoredErrorCodes), error)
               == std::cend(ignoredErrorCodes))
            {
                using namespace std::literals;
                rtCheck<TApi, TThrow>(error, ("'"s + std::string(cmd) + "' returned error "s).c_str(), file, line);
            }
            else
            {
                // reset the last error to avoid propagation to the next CUDA/HIP API call. Using std::ignore
                // to discard unneeded return values is recommended by the C++ core guidelines.
                std::ignore = TApi::getLastError();
            }
        }
    }

    //! CUDA/HIP runtime API last error checking with log and exception.
    template<typename TApi, bool TThrow>
    AV2P_FN_HOST inline void rtCheckLastError(char const* desc, char const* file, int const& line) noexcept(!TThrow)
    {
        typename TApi::Error_t const error(TApi::getLastError());
        rtCheck<TApi, TThrow>(error, desc, file, line);
    }
} // namespace aV2P::uniform_cuda_hip::detail

#    define AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, throw, ...)                                                      \
        do                                                                                                            \
        {                                                                                                             \
            ::aV2P::uniform_cuda_hip::detail::rtCheckLastError<TApi, throw>(                                          \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::aV2P::uniform_cuda_hip::detail::rtCheckIgnore<TApi, throw>(                                             \
                cmd,                                                                                                  \
                #cmd,                                                                                                 \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                {__VA_ARGS__});                                                                                       \
        } while(0)

//! CUDA/HIP runtime error checking with log and exception, ignoring specific error values
#    define AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...) AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, true, __VA_ARGS__)

//! CUDA/HIP runtime error checking with log and exception.
#    define AV2P_UNIFORM_CUDA_HIP_RT_CHECK(cmd) AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, true, )

//! CUDA/HIP runtime error checking with log and exception, ignoring specific error values
#    define AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                                  \
        AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, false, __VA_ARGS__)

//! CUDA/HIP runtime error checking with log.
#    define AV2P_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(cmd) AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, false, )
#endif
