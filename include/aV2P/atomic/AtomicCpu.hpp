/* Copyright 2025 Andrea Bocci, Felice Pantaleo
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(AV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLED)                    \
    || defined(AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(AV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLED)                   \
    || defined(AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED)

#    include "aV2P/atomic/AtomicAtomicRef.hpp"
#    include "aV2P/atomic/AtomicStdLibLock.hpp"

namespace aV2P
{
#    ifndef AV2P_DISABLE_ATOMIC_ATOMICREF
    using AtomicCpu = AtomicAtomicRef;
#    else
    using AtomicCpu = AtomicStdLibLock<16>;
#    endif // AV2P_DISABLE_ATOMIC_ATOMICREF

} // namespace aV2P

#endif
