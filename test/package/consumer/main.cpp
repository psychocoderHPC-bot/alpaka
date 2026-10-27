// SPDX-License-Identifier: MPL-2.0
#include <aV2P/alpaka.hpp>

#include <iostream>

int main()
{
    static_assert(AV2P_VERSION_MAJOR == 2);
    std::cout << "aV2P " << AV2P_VERSION_MAJOR << "." << AV2P_VERSION_MINOR << "." << AV2P_VERSION_PATCH << "\n";
    return 0;
}
