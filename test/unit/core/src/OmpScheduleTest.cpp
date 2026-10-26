/* Copyright 2022 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/OmpSchedule.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>

TEST_CASE("ompScheduleDefaultConstructor", "[core]")
{
    std::ignore = aV2P::omp::Schedule{};
}

TEST_CASE("ompScheduleConstructor", "[core]")
{
    std::ignore = aV2P::omp::Schedule{aV2P::omp::Schedule::Static, 5};
    std::ignore = aV2P::omp::Schedule{aV2P::omp::Schedule::Guided};
}

TEST_CASE("ompScheduleConstexprConstructor", "[core]")
{
    std::ignore = aV2P::omp::Schedule{aV2P::omp::Schedule::Dynamic};
}

TEST_CASE("ompGetSchedule", "[core]")
{
    std::ignore = aV2P::omp::getSchedule();
}

TEST_CASE("ompSetSchedule", "[core]")
{
    auto const expectedSchedule = aV2P::omp::Schedule{aV2P::omp::Schedule::Dynamic, 3};
    aV2P::omp::setSchedule(expectedSchedule);
    // The check makes sense only when this feature is supported
#if defined _OPENMP && _OPENMP >= 200805 && defined AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    auto const actualSchedule = aV2P::omp::getSchedule();
    REQUIRE(expectedSchedule.kind == actualSchedule.kind);
    REQUIRE(expectedSchedule.chunkSize == actualSchedule.chunkSize);
#endif
}

TEST_CASE("ompSetNoSchedule", "[core]")
{
    auto const expectedSchedule = aV2P::omp::Schedule{aV2P::omp::Schedule::Guided, 2};
    aV2P::omp::setSchedule(expectedSchedule);
    auto const noSchedule = aV2P::omp::Schedule{aV2P::omp::Schedule::NoSchedule};
    aV2P::omp::setSchedule(noSchedule);
    // The check makes sense only when this feature is supported
#if defined _OPENMP && _OPENMP >= 200805 && defined AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    auto const actualSchedule = aV2P::omp::getSchedule();
    REQUIRE(expectedSchedule.kind == actualSchedule.kind);
    REQUIRE(expectedSchedule.chunkSize == actualSchedule.chunkSize);
#endif
}
