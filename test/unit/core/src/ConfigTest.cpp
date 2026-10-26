/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/Config.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

TEST_CASE("printDefines", "[core]")
{
#if AV2P_LANG_CUDA
    std::cout << "AV2P_LANG_CUDA:" << AV2P_LANG_CUDA << std::endl;
#endif
#if AV2P_LANG_HIP
    std::cout << "AV2P_LANG_HIP:" << AV2P_LANG_HIP << std::endl;
#endif
#if AV2P_ARCH_PTX
    std::cout << "AV2P_ARCH_PTX:" << AV2P_ARCH_PTX << std::endl;
#endif
#if AV2P_COMP_NVCC
    std::cout << "AV2P_COMP_NVCC:" << AV2P_COMP_NVCC << std::endl;
#endif
#if AV2P_COMP_HIP
    std::cout << "AV2P_COMP_HIP:" << AV2P_COMP_HIP << std::endl;
#endif
#if AV2P_COMP_CLANG
    std::cout << "AV2P_COMP_CLANG:" << AV2P_COMP_CLANG << std::endl;
#endif
#if AV2P_COMP_GNUC
    std::cout << "AV2P_COMP_GNUC:" << AV2P_COMP_GNUC << std::endl;
#endif
#if AV2P_COMP_CLANG_CUDA
    std::cout << "AV2P_COMP_CLANG_CUDA:" << AV2P_COMP_CLANG_CUDA << std::endl;
#endif
}

TEST_CASE("configVersionMacros", "")
{
    static_assert(AV2P_VERSION_NUMBER(2025, 2, 1) == 202'500'200'001);
    // to long major version should be cut to 4 digits
    static_assert(AV2P_VERSION_NUMBER(12025, 2, 1) == 202'500'200'001);
    static_assert(AV2P_VERSION_NUMBER(12025, 2, 1) == AV2P_VERSION_NUMBER(2025, 2, 1));

    static_assert(AV2P_VERSION_NUMBER_NOT_AVAILABLE == 0000000000000);
    static_assert(AV2P_VERSION_NUMBER_NOT_AVAILABLE == AV2P_VERSION_NUMBER(0, 0, 0));

    static_assert(AV2P_YYYYMMDD_TO_VERSION(20'250'201) == 202'500'200'001);
    static_assert(AV2P_YYYYMMDD_TO_VERSION(20'250'201) == AV2P_VERSION_NUMBER(2025, 2, 1));
    // to long major version should be cut to 4 digits
    static_assert(AV2P_YYYYMMDD_TO_VERSION(120'250'201) == AV2P_VERSION_NUMBER(2025, 2, 1));

    static_assert(AV2P_YYYYMM_TO_VERSION(202502) == 202'500'200'000);
    static_assert(AV2P_YYYYMM_TO_VERSION(202502) == AV2P_VERSION_NUMBER(2025, 2, 0));
    // to long major version should be cut to 4 digits
    static_assert(AV2P_YYYYMM_TO_VERSION(1'202'502) == AV2P_VERSION_NUMBER(2025, 2, 0));

    static_assert(AV2P_VVRRP_TO_VERSION(12081) == 1'200'800'001);
    static_assert(AV2P_VVRRP_TO_VERSION(12081) == AV2P_VERSION_NUMBER(12, 8, 1));
    // to long major version should be cut to 4 digits
    static_assert(AV2P_VVRRP_TO_VERSION(12'345'081) == AV2P_VERSION_NUMBER(2345, 8, 1));

    static_assert(AV2P_VRP_TO_VERSION(751) == 700'500'001);
    static_assert(AV2P_VRP_TO_VERSION(751) == AV2P_VERSION_NUMBER(7, 5, 1));
    // to long major version should be cut to 4 digits
    static_assert(AV2P_VRP_TO_VERSION(1'234'567) == AV2P_VERSION_NUMBER(2345, 6, 7));

    static_assert(AV2P_VRRPP_TO_VERSION(120503) == 1'200'500'003);
    static_assert(AV2P_VRRPP_TO_VERSION(120503) == AV2P_VERSION_NUMBER(12, 5, 3));
    // to long major version should be cut to 4 digits
    static_assert(AV2P_VRRPP_TO_VERSION(123'451'513) == AV2P_VERSION_NUMBER(2345, 15, 13));
}
