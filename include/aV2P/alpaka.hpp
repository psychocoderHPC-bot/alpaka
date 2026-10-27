/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Jan Stephan, Antonio Di Pilato, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Include the whole library.

// alpaka 3 implementation
#include <alpaka/alpaka.hpp>

// version number
#include "aV2P/version.hpp"
// acc frontend
#include "aV2P/acc/AccDevProps.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/acc/Traits.hpp"
// algo
#include "aV2P/algo/Transform.hpp"
// atomic
// In alpaka 2.2 AtomicAtomicRef was exposed transitively through its CPU atomic path, so the umbrella made it visible.
#include "aV2P/atomic/AtomicAtomicRef.hpp"
#include "aV2P/atomic/Op.hpp"
#include "aV2P/atomic/Traits.hpp"
// block
#include "aV2P/block/shared/dyn/Traits.hpp"
#include "aV2P/block/shared/st/Traits.hpp"
#include "aV2P/block/sync/Traits.hpp"
// core
#include "aV2P/core/Align.hpp"
#include "aV2P/core/AlignedAlloc.hpp"
#include "aV2P/core/Assert.hpp"
#include "aV2P/core/BarrierThread.hpp"
#include "aV2P/core/ClipCast.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Debug.hpp"
#include "aV2P/core/Decay.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/core/OmpSchedule.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/core/RemoveRestrict.hpp"
#include "aV2P/core/RuntimeMacros.hpp"
#include "aV2P/core/ThreadPool.hpp"
#include "aV2P/core/Unreachable.hpp"
#include "aV2P/core/Unroll.hpp"
#include "aV2P/core/Utility.hpp"
#include "aV2P/core/Vectorize.hpp"
// dev frontend
#include "aV2P/dev/Traits.hpp"
// dim
#include "aV2P/dim/DimArithmetic.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/dim/Traits.hpp"
// event frontend
#include "aV2P/event/Traits.hpp"
// exec
#include "aV2P/exec/ElementIndex.hpp"
#include "aV2P/exec/IndependentElements.hpp"
#include "aV2P/exec/Once.hpp"
#include "aV2P/exec/UniformElements.hpp"
// extent
#include "aV2P/extent/Traits.hpp"
// grid
#include "aV2P/grid/GridSyncAcc.hpp"
#include "aV2P/grid/GridSyncBarrierCpuOmp.hpp"
#include "aV2P/grid/GridSyncBarrierCpuThread.hpp"
#include "aV2P/grid/GridSyncCpuTbbBlocks.hpp"
#include "aV2P/grid/GridSyncGpuCudaHip.hpp"
#include "aV2P/grid/GridSyncNoOp.hpp"
#include "aV2P/grid/Traits.hpp"
// idx
#include "aV2P/idx/Accessors.hpp"
#include "aV2P/idx/MapIdx.hpp"
#include "aV2P/idx/Traits.hpp"
// kernel frontend
#include "aV2P/kernel/KernelFunctionAttributes.hpp"
#include "aV2P/kernel/Traits.hpp"
// math
#include "aV2P/math/Complex.hpp"
#include "aV2P/math/FloatEqualExact.hpp"
#include "aV2P/math/Traits.hpp"
// mem
#include "aV2P/mem/alloc/AllocCpuAligned.hpp"
#include "aV2P/mem/alloc/AllocCpuNew.hpp"
#include "aV2P/mem/alloc/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/fence/Traits.hpp"
#include "aV2P/mem/global/DeviceGlobalCpu.hpp"
#include "aV2P/mem/global/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/mem/view/ViewConst.hpp"
#include "aV2P/mem/view/ViewPlainPtr.hpp"
#include "aV2P/mem/view/ViewStdArray.hpp"
#include "aV2P/mem/view/ViewStdSpan.hpp"
#include "aV2P/mem/view/ViewStdVector.hpp"
#include "aV2P/mem/view/ViewSubView.hpp"
// meta
#include "aV2P/meta/Apply.hpp"
#include "aV2P/meta/CartesianProduct.hpp"
#include "aV2P/meta/Concatenate.hpp"
#include "aV2P/meta/DependentFalseType.hpp"
#include "aV2P/meta/Filter.hpp"
#include "aV2P/meta/Fold.hpp"
#include "aV2P/meta/ForEachType.hpp"
#include "aV2P/meta/Functional.hpp"
#include "aV2P/meta/IntegerSequence.hpp"
#include "aV2P/meta/Integral.hpp"
#include "aV2P/meta/IsArrayOrVector.hpp"
#include "aV2P/meta/IsStrictBase.hpp"
#include "aV2P/meta/NdLoop.hpp"
#include "aV2P/meta/NonZero.hpp"
#include "aV2P/meta/Set.hpp"
#include "aV2P/meta/Transform.hpp"
#include "aV2P/meta/TypeListOps.hpp"
// offset
#include "aV2P/offset/Traits.hpp"
// platform frontend
#include "aV2P/platform/Traits.hpp"
// queue frontend
#include "aV2P/queue/Properties.hpp"
#include "aV2P/queue/Traits.hpp"
// rand
#include "aV2P/rand/RandDefault.hpp"
#include "aV2P/rand/RandPhilox.hpp"
#include "aV2P/rand/RandStdLib.hpp"
#include "aV2P/rand/Traits.hpp"
// traits
#include "aV2P/traits/Traits.hpp"
// wait
#include "aV2P/wait/Traits.hpp"
// workdiv
#include "aV2P/workdiv/Traits.hpp"
#include "aV2P/workdiv/WorkDivHelpers.hpp"
#include "aV2P/workdiv/WorkDivMembers.hpp"
// vec
#include "aV2P/vec/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
// warp frontend
#include "aV2P/warp/Traits.hpp"

// aV2P -> alpaka 3 adapter layer (must come after the frontend declarations)
#include "aV2P/adapter/Adapter.hpp"
// accelerator tag enablement depends on the TagToAcc specializations from the adapter
#include "aV2P/acc/TagAccIsEnabled.hpp"
