/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Matthias Werner, Jan Stephan, Andrea Bocci, Bernhard
 * Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/core/Cuda.hpp"
#include "aV2P/core/Hip.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    namespace gb
    {
        //! The CUDA/HIP accelerator ND index provider.
        template<typename TDim, typename TIdx>
        class IdxGbUniformCudaHipBuiltIn
            : public interface::Implements<ConceptIdxGb, IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
        };
    } // namespace gb

#    if !defined(AV2P_HOST_ONLY)

#        if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA
#            error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP
#            error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        //! The GPU CUDA/HIP accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The GPU CUDA/HIP accelerator grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            __device__ static auto getIdx(gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx> const& /* idx */, TWorkDiv const&)
                -> Vec<TDim, TIdx>
            {
#        if defined(AV2P_ACC_GPU_CUDA_ENABLED)
                return castVec<TIdx>(getOffsetVecEnd<TDim>(blockIdx));
#        else
                return getOffsetVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipBlockIdx_z),
                    static_cast<TIdx>(hipBlockIdx_y),
                    static_cast<TIdx>(hipBlockIdx_x)));
#        endif
            }
        };

        //! The GPU CUDA/HIP accelerator grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait

#    endif

} // namespace aV2P

#endif
