
#pragma once

#include "aV2P/dev/Traits.hpp"

#include <string>
#include <vector>

namespace aV2P
{

    struct DeviceProperties
    {
        std::string name;
        std::size_t totalGlobalMem;
        std::vector<std::size_t> warpSizes;
        std::size_t preferredWarpSize;
    };
} // namespace aV2P
