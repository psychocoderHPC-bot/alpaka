/* Copyright (c) 2020 , Argonne National Laboratory   (Zheming Jin)
 * Copyright (c) 2020-, Oak Ridge National Laboratory   (Zheming Jin)
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file The file is providing some helpers for the HACCmk implementation shared between the original and the alpaka
 * implementation.
 */

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>

namespace hacc
{
    void haccmk_gold(
        int count1,
        float xxi,
        float yyi,
        float zzi,
        float fsrrmax2,
        float mp_rsm2,
        auto xx1,
        auto yy1,
        auto zz1,
        auto mass1,
        auto dxi,
        auto dyi,
        auto dzi)
    {
        float const ma0 = 0.269327f, ma1 = -0.0750978f, ma2 = 0.0114808f, ma3 = -0.00109313f, ma4 = 0.0000605491f,
                    ma5 = -0.00000147177f;


        float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

        xi = 0.f;
        yi = 0.f;
        zi = 0.f;

        for(int j = 0; j < count1; j++)
        {
            dxc = xx1[j] - xxi;
            dyc = yy1[j] - yyi;
            dzc = zz1[j] - zzi;

            r2 = dxc * dxc + dyc * dyc + dzc * dzc;

            if(r2 < fsrrmax2)
                m = mass1[j];
            else
                m = 0.f;

            f = r2 + mp_rsm2;
            f = m * (1.f / (f * sqrtf(f)) - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));

            xi = xi + f * dxc;
            yi = yi + f * dyc;
            zi = zi + f * dzc;
        }

        *dxi = xi;
        *dyi = yi;
        *dzi = zi;
    }

    int verify(int n2, auto vx2, auto vy2, auto vz2, auto vx2_hw, auto vy2_hw, auto vz2_hw)
    {
        int error = EXIT_SUCCESS;
        float const eps = 1.0f;
        for(int i = 0; i < n2; i++)
        {
            if(fabsf(vx2[i] - vx2_hw[i]) > eps)
            {
                printf("error at vx2[%d] %f %f\n", i, vx2[i], vx2_hw[i]);
                error = EXIT_FAILURE;
                break;
            }
            if(fabsf(vy2[i] - vy2_hw[i]) > eps)
            {
                printf("error at vy2[%d]: %f %f\n", i, vy2[i], vy2_hw[i]);
                error = EXIT_FAILURE;
                break;
            }
            if(fabsf(vz2[i] - vz2_hw[i]) > eps)
            {
                printf("error at vz2[%d]: %f %f\n", i, vz2[i], vz2_hw[i]);
                error = EXIT_FAILURE;
                break;
            }
        }
        printf("%s\n", error ? "Verification FAILED" : "Verification PASSED");
        return error;
    }

    /** Accurate reference for verification.
     *
     * The original gold function accumulates in single precision with a linear summation order, so its own rounding
     * error grows with the number of terms. Accumulating in double yields a reference close to the exact result, so
     * accelerators using a different reduction order (e.g. a tree reduction) can be judged against it with a
     * magnitude-aware tolerance.
     */
    inline void haccmk_gold_double(
        int count1,
        float xxi,
        float yyi,
        float zzi,
        float fsrrmax2,
        float mp_rsm2,
        auto xx1,
        auto yy1,
        auto zz1,
        auto mass1,
        double* dxi,
        double* dyi,
        double* dzi)
    {
        double const ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491,
                     ma5 = -0.00000147177;

        double xi = 0.0;
        double yi = 0.0;
        double zi = 0.0;

        for(int j = 0; j < count1; j++)
        {
            double dxc = static_cast<double>(xx1[j]) - static_cast<double>(xxi);
            double dyc = static_cast<double>(yy1[j]) - static_cast<double>(yyi);
            double dzc = static_cast<double>(zz1[j]) - static_cast<double>(zzi);

            double r2 = dxc * dxc + dyc * dyc + dzc * dzc;

            /* The kernel decides the cutoff with fp32 arithmetic. A double-rounded r2 can land on the other side
             * of fsrrmax2 than the float value the kernel uses, which flips an O(mass) contribution on or off.
             * Replicate the kernel's fp32 decision so the reference measures summation error instead of that
             * ambiguity. */
            float const dxcF = static_cast<float>(xx1[j]) - static_cast<float>(xxi);
            float const dycF = static_cast<float>(yy1[j]) - static_cast<float>(yyi);
            float const dzcF = static_cast<float>(zz1[j]) - static_cast<float>(zzi);
            float const r2f = dxcF * dxcF + dycF * dycF + dzcF * dzcF;

            double m = (r2f < fsrrmax2) ? static_cast<double>(mass1[j]) : 0.0;

            double f = r2 + static_cast<double>(mp_rsm2);
            f = m * (1.0 / (f * std::sqrt(f)) - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));

            xi = xi + f * dxc;
            yi = yi + f * dyc;
            zi = zi + f * dzc;
        }

        *dxi = xi;
        *dyi = yi;
        *dzi = zi;
    }

    /** Verify the accelerator result against a high precision reference.
     *
     * The comparison uses the combined criterion
     * |ref - test| <= atol + rtol * max(|ref|, |test|) per entry and component. A relative part is required because
     * the accelerator may reduce in a different order than the sequential reference, so the absolute error scales
     * with the magnitude of the result. With the fp32 cutoff decision mirrored by the reference the residual is
     * dominated by summation rounding of the kernel: for the 60000/120000 element cases a float SIMD-lane
     * reduction stays within ~2e-6 relative error, so 1e-5 keeps a margin of at least 5 while rejecting
     * regressions above ~1e-5 (an absolute tolerance of 1.0 or a relative tolerance of 1e-3 only detects
     * ~1e-3).
     *
     * @return EXIT_SUCCESS if all n entries of the three velocity components pass.
     */
    template<typename TRefX, typename TRefY, typename TRefZ, typename THwX, typename THwY, typename THwZ>
    int verifyRelative(
        int n,
        TRefX const& vx2,
        TRefY const& vy2,
        TRefZ const& vz2,
        THwX const& vx2_hw,
        THwY const& vy2_hw,
        THwZ const& vz2_hw,
        double atol = 1e-3,
        double rtol = 1e-5)
    {
        double maxAbs[3] = {0.0, 0.0, 0.0};
        double maxRel[3] = {0.0, 0.0, 0.0};
        bool reported[3] = {false, false, false};
        char const* names[3] = {"vx2", "vy2", "vz2"};
        int error = EXIT_SUCCESS;

        for(int i = 0; i < n; i++)
        {
            double const refs[3]
                = {static_cast<double>(vx2[i]), static_cast<double>(vy2[i]), static_cast<double>(vz2[i])};
            double const tests[3]
                = {static_cast<double>(vx2_hw[i]), static_cast<double>(vy2_hw[i]), static_cast<double>(vz2_hw[i])};
            for(int c = 0; c < 3; c++)
            {
                // A NaN or Inf slips through the magnitude comparison below, so reject non-finite values explicitly.
                if(!std::isfinite(refs[c]) || !std::isfinite(tests[c]))
                {
                    if(!reported[c])
                    {
                        printf(
                            "error at %s[%d] reference %f kernel %f (non-finite value)\n",
                            names[c],
                            i,
                            refs[c],
                            tests[c]);
                        reported[c] = true;
                    }
                    error = EXIT_FAILURE;
                    continue;
                }
                double const diff = std::fabs(refs[c] - tests[c]);
                double const scale = std::max(std::fabs(refs[c]), std::fabs(tests[c]));
                double const rel = scale > 0.0 ? diff / scale : 0.0;
                maxAbs[c] = std::max(maxAbs[c], diff);
                maxRel[c] = std::max(maxRel[c], rel);
                if(diff > atol + rtol * scale)
                {
                    if(!reported[c])
                    {
                        printf(
                            "error at %s[%d] reference %f kernel %f (abs %.3e rel %.3e)\n",
                            names[c],
                            i,
                            refs[c],
                            tests[c],
                            diff,
                            rel);
                        reported[c] = true;
                    }
                    error = EXIT_FAILURE;
                }
            }
        }

        printf("max absolute diff: vx2 %.3e vy2 %.3e vz2 %.3e\n", maxAbs[0], maxAbs[1], maxAbs[2]);
        printf("max relative error: vx2 %.3e vy2 %.3e vz2 %.3e\n", maxRel[0], maxRel[1], maxRel[2]);
        printf("%s\n", error ? "Verification FAILED" : "Verification PASSED");
        return error;
    }

} // namespace hacc
