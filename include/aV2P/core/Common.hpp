/* Copyright 2024 Axel Hübl, Benjamin Worpitz, Matthias Werner, Jan Stephan, René Widera, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/core/Debug.hpp"

// Boost.Uuid errors with VS2017 when intrin.h is not included
#if defined(_MSC_VER) && _MSC_VER >= 1910
#    include <intrin.h>
#endif

#if AV2P_LANG_HIP
#    if AV2P_COMP_HIP >= AV2P_VERSION_NUMBER(6, 2, 0) && AV2P_COMP_HIP < AV2P_VERSION_NUMBER(7, 0, 0)
#        define HIP_ENABLE_WARP_SYNC_BUILTINS
#    endif
// HIP defines some keywords like __forceinline__ in header files.
#    include <hip/hip_runtime.h>
#endif

//! All functions that can be used on an accelerator have to be attributed with AV2P_FN_ACC or AV2P_FN_HOST_ACC.
//!
//! \code{.cpp}
//! Usage:
//! AV2P_FN_ACC
//! auto add(std::int32_t a, std::int32_t b)
//! -> std::int32_t;
//! \endcode
//! @{
#if AV2P_LANG_CUDA || AV2P_LANG_HIP
// ONLY_MODE is a 2.2 compatibility validation: it restricts the enabled back-ends, it must not
// change the function attributes.  Native alpaka 3 decides whether a kernel can be launched by
// checking that the functor is invocable on the host (`KernelBundle::operator()`), so a
// `__device__`-only functor is rejected.  Keep `__device__ __host__` in every mode.
#    define AV2P_FN_ACC __device__ __host__
#    define AV2P_FN_HOST_ACC __device__ __host__
#    define AV2P_FN_HOST __host__
#else
#    define AV2P_FN_ACC
#    define AV2P_FN_HOST_ACC
#    define AV2P_FN_HOST
#endif
//! @}

//! All functions marked with AV2P_FN_ACC or AV2P_FN_HOST_ACC that are exported to / imported from different
//! translation units have to be attributed with AV2P_FN_EXTERN. Note that this needs to be applied to both the
//! declaration and the definition.
//!
//! Usage:
//! AV2P_FN_ACC AV2P_FN_EXTERN auto add(std::int32_t a, std::int32_t b) -> std::int32_t;
//!
//! Warning: If this is used together with the SYCL back-end make sure that your SYCL runtime supports generic
//! address spaces. Otherwise it is forbidden to use pointers as parameter or return type for functions marked
//! with AV2P_FN_EXTERN.
#ifdef AV2P_ACC_SYCL_ENABLED
/*
   This is required by the SYCL standard, section 5.10.1 "SYCL functions and member functions linkage":

   The default behavior in SYCL applications is that all the definitions and declarations of the functions and member
   functions are available to the SYCL compiler, in the same translation unit. When this is not the case, all the
   symbols that need to be exported to a SYCL library or from a C++ library to a SYCL application need to be defined
   using the macro: SYCL_EXTERNAL.
*/
#    define AV2P_FN_EXTERN SYCL_EXTERNAL
#else
#    define AV2P_FN_EXTERN
#endif

//! Disable nvcc warning:
//! 'calling a __host__ function from __host__ __device__ function.'
//! Usage:
//! AV2P_NO_HOST_ACC_WARNING
//! AV2P_FN_HOST_ACC function_declaration()
//! WARNING: Only use this method if there is no other way.
//! Most cases can be solved by #if AV2P_ARCH_PTX or #if AV2P_LANG_CUDA.
#if(AV2P_LANG_CUDA && !AV2P_COMP_CLANG_CUDA)
#    if AV2P_COMP_MSVC
#        define AV2P_NO_HOST_ACC_WARNING __pragma(hd_warning_disable)
#    else
#        define AV2P_NO_HOST_ACC_WARNING _Pragma("hd_warning_disable")
#    endif
#else
#    define AV2P_NO_HOST_ACC_WARNING
#endif

//! Macro defining the inline function attribute.
//!
//! The macro should stay on the left hand side of keywords, e.g. 'static', 'constexpr', 'explicit' or the return type.
#if AV2P_LANG_CUDA || AV2P_LANG_HIP
#    define AV2P_FN_INLINE __forceinline__
#elif AV2P_COMP_MSVC
// TODO: With C++20 [[msvc::forceinline]] can be used.
#    define AV2P_FN_INLINE __forceinline
#else
// For gcc, clang, and clang-based compilers like Intel icpx
#    define AV2P_FN_INLINE [[gnu::always_inline]] inline
#endif

//! This macro defines a variable lying in global accelerator device memory.
//!
//! Example:
//!   AV2P_STATIC_ACC_MEM_GLOBAL aV2P::DevGlobal<TAcc, int> variable;
//!
//! Those variables behave like ordinary variables when used in file-scope,
//! but inside kernels the get() method must be used to access the variable.
//! They are declared inline to resolve to a single instance across multiple
//! translation units.
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
//!
//! \attention It is not allowed to initialize the variable together with the declaration.
//!            To initialize the variable aV2P::memcpy must be used.
//! \code{.cpp}
//! AV2P_STATIC_ACC_MEM_GLOBAL aV2P::DevGlobal<TAcc, int> foo;
//!
//! struct DeviceMemoryKernel
//! {
//!    AV2P_NO_HOST_ACC_WARNING
//!    template<typename TAcc>
//!    AV2P_FN_ACC void operator()(TAcc const& acc) const
//!    {
//!      auto a = foo<TAcc>.get();
//!    }
//!  }
//!
//! void initFoo() {
//!     auto extent = aV2P::Vec<aV2P::DimInt<1u>, size_t>{1};
//!     int initialValue = 42;
//!     aV2P::ViewPlainPtr<DevHost, int, aV2P::DimInt<1u>, size_t> bufHost(&initialValue, devHost, extent);
//!     aV2P::memcpy(queue, foo<Acc>, bufHost, extent);
//! }
//! \endcode
#if((AV2P_LANG_CUDA && AV2P_COMP_CLANG_CUDA) || (AV2P_LANG_CUDA && AV2P_COMP_NVCC && AV2P_ARCH_PTX)         \
    || AV2P_LANG_HIP)
#    if defined(__CUDACC_RDC__) || defined(__CLANG_RDC__)
#        define AV2P_STATIC_ACC_MEM_GLOBAL                                                                          \
            template<typename TAcc>                                                                                   \
            __device__ inline
#    else
#        define AV2P_STATIC_ACC_MEM_GLOBAL                                                                          \
            template<typename TAcc>                                                                                   \
            __device__ static
#    endif
#else
#    define AV2P_STATIC_ACC_MEM_GLOBAL                                                                              \
        template<typename TAcc>                                                                                       \
        inline
#endif

//! This macro defines a variable lying in constant accelerator device memory.
//!
//! Example:
//!   AV2P_STATIC_ACC_MEM_CONSTANT aV2P::DevGlobal<TAcc, const int> variable;
//!
//! Those variables behave like ordinary variables when used in file-scope,
//! but inside kernels the get() method must be used to access the variable.
//! They are declared inline to resolve to a single instance across multiple
//! translation units.
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
//!
//! \attention It is not allowed to initialize the variable together with the declaration.
//!            To initialize the variable aV2P::memcpy must be used.
//! \code{.cpp}
//! AV2P_STATIC_ACC_MEM_CONSTANT aV2P::DevGlobal<TAcc, const int> foo;
//!
//! struct DeviceMemoryKernel
//! {
//!    AV2P_NO_HOST_ACC_WARNING
//!    template<typename TAcc>
//!    AV2P_FN_ACC void operator()(TAcc const& acc) const
//!    {
//!      auto a = foo<TAcc>.get();
//!    }
//!  }
//!
//! void initFoo() {
//!     auto extent = aV2P::Vec<aV2P::DimInt<1u>, size_t>{1};
//!     int initialValue = 42;
//!     aV2P::ViewPlainPtr<DevHost, int, aV2P::DimInt<1u>, size_t> bufHost(&initialValue, devHost, extent);
//!     aV2P::memcpy(queue, foo<Acc>, bufHost, extent);
//! }
//! \endcode
#if((AV2P_LANG_CUDA && AV2P_COMP_CLANG_CUDA) || (AV2P_LANG_CUDA && AV2P_COMP_NVCC && AV2P_ARCH_PTX)         \
    || AV2P_LANG_HIP)
#    if defined(__CUDACC_RDC__) || defined(__CLANG_RDC__)
#        define AV2P_STATIC_ACC_MEM_CONSTANT                                                                        \
            template<typename TAcc>                                                                                   \
            __constant__ inline
#    else
#        define AV2P_STATIC_ACC_MEM_CONSTANT                                                                        \
            template<typename TAcc>                                                                                   \
            __constant__ static
#    endif
#else
#    define AV2P_STATIC_ACC_MEM_CONSTANT                                                                            \
        template<typename TAcc>                                                                                       \
        inline
#endif

//! This macro disables memory optimizations for annotated device memory.
//!
//! Example:
//!   AV2P_DEVICE_VOLATILE float* ptr;
//!
//! This is useful for pointers, (shared) variables and shared memory which are used in combination with
//! the aV2P::mem_fence() function. It ensures that memory annotated with this macro will always be written directly
//! to memory (and not to a register or cache because of compiler optimizations).
#if(AV2P_LANG_CUDA && AV2P_ARCH_PTX)                                                                              \
    || (AV2P_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1)
#    define AV2P_DEVICE_VOLATILE volatile
#else
#    define AV2P_DEVICE_VOLATILE
#endif
