/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/core/Cuda.hpp"
#include "aV2P/core/Hip.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/Traits.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP accelerator work division.
    template<typename TDim, typename TIdx>
    class WorkDivUniformCudaHipBuiltIn
        : public interface::Implements<ConceptWorkDiv, WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
    {
    public:
        AV2P_FN_HOST_ACC WorkDivUniformCudaHipBuiltIn(Vec<TDim, TIdx> const& threadElemExtent)
            : m_threadElemExtent(threadElemExtent)
        {
        }

        // \TODO: Optimize! Add WorkDivUniformCudaHipBuiltInNoElems that has no member m_threadElemExtent as well as
        // AccGpuUniformCudaHipRtNoElems. Use it instead of AccGpuUniformCudaHipRt if the thread element extent is one
        // to reduce the register usage.
        Vec<TDim, TIdx> const& m_threadElemExtent;
    };

#    if !defined(AV2P_HOST_ONLY)

#        if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA
#            error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP
#            error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        //! The GPU CUDA/HIP accelerator work division dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The GPU CUDA/HIP accelerator work division idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The GPU CUDA/HIP accelerator work division grid block extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The number of blocks in each dimension of the grid.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& /* workDiv */)
                -> Vec<TDim, TIdx>
            {
#        ifdef AV2P_ACC_GPU_CUDA_ENABLED
                return castVec<TIdx>(getExtentVecEnd<TDim>(gridDim));
#        else
                return getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipGridDim_z),
                    static_cast<TIdx>(hipGridDim_y),
                    static_cast<TIdx>(hipGridDim_x)));
#        endif
            }
        };

        //! The GPU CUDA/HIP accelerator work division block thread extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The number of threads in each dimension of a block.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& /* workDiv */)
                -> Vec<TDim, TIdx>
            {
#        ifdef AV2P_ACC_GPU_CUDA_ENABLED
                return castVec<TIdx>(getExtentVecEnd<TDim>(blockDim));
#        else
                return getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipBlockDim_z),
                    static_cast<TIdx>(hipBlockDim_y),
                    static_cast<TIdx>(hipBlockDim_x)));
#        endif
            }
        };

        //! The GPU CUDA/HIP accelerator work division thread element extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Thread, unit::Elems>
        {
            //! \return The number of blocks in each dimension of the grid.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& workDiv)
                -> Vec<TDim, TIdx>
            {
                return workDiv.m_threadElemExtent;
            }
        };
    } // namespace trait

#    endif

} // namespace aV2P

#endif
