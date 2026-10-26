#pragma once

#include "aV2P/acc/Traits.hpp"
#include "aV2P/core/Unreachable.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/WorkDivMembers.hpp"

#include <type_traits>

// Create an accelerator-dependent work division for 1-dimensional kernels.
template<aV2P::concepts::Acc TAcc>
requires(aV2P::Dim<TAcc>::value == 1)
inline auto makeWorkDiv(aV2P::Idx<TAcc> blocks, aV2P::Idx<TAcc> elements)
    -> aV2P::WorkDivMembers<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>>
{
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;
    using WorkDiv = aV2P::WorkDivMembers<Dim, Idx>;
    if constexpr(aV2P::isMultiThreadAcc<TAcc>)
    {
        // On thread-parallel backends, each thread is looking at a single element:
        //   - the number of threads per block is "elements";   - the number of elements per thread is always 1.
        return WorkDiv{blocks, elements, Idx{1}};
    }
    else
    {
        // On thread-serial backends, run serially with a single thread per block:
        //   - the number of threads per block is always 1;   - the number of elements per thread is "elements".
        return WorkDiv{blocks, Idx{1}, elements};
    }

    AV2P_UNREACHABLE(WorkDiv{blocks, elements, Idx{1}});
}

// Create the accelerator-dependent workdiv for N-dimensional kernels.
template<aV2P::concepts::Acc TAcc>
inline auto makeWorkDiv(
    aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& blocks,
    aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& elements)
    -> aV2P::WorkDivMembers<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>>
{
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;
    using Vec = aV2P::Vec<Dim, Idx>;
    using WorkDiv = aV2P::WorkDivMembers<Dim, Idx>;
    if constexpr(aV2P::isMultiThreadAcc<TAcc>)
    {
        // On thread-parallel backends, each thread is looking at a single element:
        //   - the number of threads per block is "elements";   - the number of elements per thread is always 1.
        return WorkDiv{blocks, elements, Vec::ones()};
    }
    else
    {
        // On thread-serial backends, run serially with a single thread per block:
        //   - the number of threads per block is always 1;   - the number of elements per thread is "elements".
        return WorkDiv{blocks, Vec::ones(), elements};
    }

    AV2P_UNREACHABLE(WorkDiv{blocks, elements, Vec::ones()});
}
