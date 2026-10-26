/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/block/shared/st/Traits.hpp"
#include "aV2P/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp"

#include <cstddef>
#include <cstdint>

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P
{
    //! The generic SYCL shared memory allocator.
    class BlockSharedMemStGenericSycl
        : public aV2P::detail::BlockSharedMemStMemberImpl<>
        , public interface::Implements<ConceptBlockSharedSt, BlockSharedMemStGenericSycl>
    {
    public:
        BlockSharedMemStGenericSycl(sycl::local_accessor<std::byte> accessor)
            : BlockSharedMemStMemberImpl(
                reinterpret_cast<std::uint8_t*>(accessor.get_multi_ptr<sycl::access::decorated::no>().get()),
                accessor.size())
            , m_accessor{accessor}
        {
        }

    private:
        sycl::local_accessor<std::byte> m_accessor;
    };
} // namespace aV2P

namespace aV2P::trait
{
    template<typename T, std::size_t TUniqueId>
    struct DeclareSharedVar<T, TUniqueId, BlockSharedMemStGenericSycl>
    {
        static auto declareVar(BlockSharedMemStGenericSycl const& smem) -> T&
        {
            auto* data = smem.template getVarPtr<T>(TUniqueId);

            if(!data)
            {
                smem.template alloc<T>(TUniqueId);
                data = smem.template getLatestVarPtr<T>();
            }
            AV2P_ASSERT(data != nullptr);
            return *data;
        }
    };

    template<>
    struct FreeSharedVars<BlockSharedMemStGenericSycl>
    {
        static auto freeVars(BlockSharedMemStGenericSycl const&) -> void
        {
            // shared memory block data will be reused
        }
    };
} // namespace aV2P::trait

#endif
