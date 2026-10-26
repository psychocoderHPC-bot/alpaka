/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/block/shared/dyn/Traits.hpp"

#include <cstddef>

#ifdef AV2P_ACC_SYCL_ENABLED
#    include <sycl/sycl.hpp>

namespace aV2P
{
    //! The SYCL block shared memory allocator.
    class BlockSharedMemDynGenericSycl
        : public interface::Implements<ConceptBlockSharedDyn, BlockSharedMemDynGenericSycl>
    {
    public:
        using BlockSharedMemDynBase = BlockSharedMemDynGenericSycl;

        BlockSharedMemDynGenericSycl(sycl::local_accessor<std::byte> accessor) : m_accessor{accessor}
        {
        }

        sycl::local_accessor<std::byte> m_accessor;
    };
} // namespace aV2P

namespace aV2P::trait
{
    template<typename T>
    struct GetDynSharedMem<T, BlockSharedMemDynGenericSycl>
    {
        static auto getMem(BlockSharedMemDynGenericSycl const& shared) -> T*
        {
            return reinterpret_cast<T*>(shared.m_accessor.get_multi_ptr<sycl::access::decorated::no>().get());
        }
    };
} // namespace aV2P::trait

#endif
