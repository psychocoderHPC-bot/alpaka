// SPDX-License-Identifier: MPL-2.0
#include <aV2P/alpaka.hpp>

#include <iostream>

int main()
{
    static_assert(AV2P_VERSION_MAJOR == 2);
    std::cout << "add_subdirectory consumer ok\n";
    return 0;
}
