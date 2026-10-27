// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>

#include <aV2P/alpaka.hpp>

#include <cstddef>

int main()
{
    static_assert(AV2P_VERSION_MAJOR == 2);
    static_assert(ALPAKA_VERSION_MAJOR == 3);

    using LegacyDim = aV2P::DimInt<1u>;
    static_assert(LegacyDim::value == 1u);

    alpaka::api::Host nativeApi{};
    (void) nativeApi;
    return 0;
}
