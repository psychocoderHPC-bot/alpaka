/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber,
 *                Aurora Perego, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

// no pragma once, because the header defines alias namespaces

#ifdef AV2P_USE_MDSPAN
#    ifdef AV2P_HAS_STD_MDSPAN
//       mdspan from the standard library
#        include <mdspan>
#        include <version>

namespace aV2P::experimental
{
    // Import C++23 mdspan into aV2P::experimental namespace.
    // See https://wg21.link/P0009R18 .
    using ::std::default_accessor;
    using ::std::dextents;
    using ::std::extents;
    using ::std::layout_left;
    using ::std::layout_right;
    using ::std::layout_stride;
    using ::std::mdspan;
#        ifdef __cpp_lib_submdspan
    // Import C++26 submdspan into aV2P::experimental namespace.
    // See https://wg21.link/P2630R4 .
    using ::std::full_extent;
    using ::std::submdspan;
#        endif
} // namespace aV2P::experimental

#    else

#        ifdef AV2P_ACC_SYCL_ENABLED
//           do not expose the macro definition of printf
#            pragma push_macro("printf")
#            ifdef printf
#                undef printf
#            endif
#        endif // AV2P_ACC_SYCL_ENABLED

//       mdspan from the Kokkos reference implementation
#        define MDSPAN_IMPL_STANDARD_NAMESPACE aV2P::experimental
#        include <experimental/mdspan>

#        ifdef AV2P_ACC_SYCL_ENABLED
//           restore the macro definition of printf
#            pragma pop_macro("printf")
#        endif // AV2P_ACC_SYCL_ENABLED
#    endif
#endif
