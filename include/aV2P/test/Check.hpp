/* Copyright 2023 Benjamin Worpitz, Jan Stephan, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Sycl.hpp"

#include <cstdio>

#define AV2P_CHECK(success, expression)                                                                             \
    do                                                                                                                \
    {                                                                                                                 \
        if(!(expression))                                                                                             \
        {                                                                                                             \
            printf("AV2P_CHECK failed because '!(%s)'\n", #expression);                                             \
            success = false;                                                                                          \
        }                                                                                                             \
    } while(0)
