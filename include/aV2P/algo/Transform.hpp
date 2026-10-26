/* Copyright 2025 Andrea Bocci, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/elem/Traits.hpp"
#include "aV2P/exec/UniformElements.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/WorkDivHelpers.hpp"

#include <iterator>
#include <type_traits>

namespace aV2P
{

    namespace detail
    {

        template<typename TFn>
        struct TransformKernel
        {
            TFn fn;

            template<typename TAcc, typename T>
            AV2P_FN_ACC void operator()(TAcc const& acc, T const* in_ptr, T* out_ptr, aV2P::Idx<TAcc> size) const
            {
                static_assert(std::is_invocable_r_v<T, TFn, T> or std::is_invocable_r_v<T, TFn, TAcc const&, T>);

                static_assert(aV2P::Dim<TAcc>::value == 1u);
                using Idx = aV2P::Idx<TAcc>;

                for(Idx i : aV2P::uniformElements(acc, size))
                {
                    if constexpr(std::is_invocable_r_v<T, TFn, T>)
                    {
                        // std::is_invocable_r_v<T, TFn, T>
                        out_ptr[i] = fn(in_ptr[i]);
                    }
                    else
                    {
                        // std::is_invocable_r_v<T, TFn, TAcc const&, T>
                        out_ptr[i] = fn(acc, in_ptr[i]);
                    }
                }
            }
        };

        template<typename TFn>
        struct TransformKernelND
        {
            TFn fn;

            template<typename TAcc, typename T>
            AV2P_FN_ACC void operator()(
                TAcc const& acc,
                T const* in_ptr,
                aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> in_pithces,
                T* out_ptr,
                aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> out_pitches,
                aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> in_size) const
            {
                static_assert(std::is_invocable_r_v<T, TFn, T> or std::is_invocable_r_v<T, TFn, TAcc const&, T>);

                using Dim = aV2P::Dim<TAcc>;
                using Idx = aV2P::Idx<TAcc>;
                using Vec = aV2P::Vec<Dim, Idx>;

                for(Vec idx : aV2P::uniformElementsND(acc, in_size))
                {
                    auto p_in = reinterpret_cast<T const*>(
                        reinterpret_cast<uintptr_t>(in_ptr) + static_cast<uintptr_t>((idx * in_pithces).sum()));
                    auto p_out = reinterpret_cast<T*>(
                        reinterpret_cast<uintptr_t>(out_ptr) + static_cast<uintptr_t>((idx * out_pitches).sum()));
                    if constexpr(std::is_invocable_r_v<T, TFn, T>)
                    {
                        // std::is_invocable_r_v<T, TFn, T>
                        *p_out = fn(*p_in);
                    }
                    else
                    {
                        // std::is_invocable_r_v<T, TFn, TAcc const&, T>
                        *p_out = fn(acc, *p_in);
                    }
                }
            }
        };

    } // namespace detail

    /*
     * Applies asynchronously the given function `fn` to the elements of the input range starting at `in`,
     * and stores the result in the semi-open output range [`out_begin`,`out_end`), using the accelerator
     * back-end identified by `Tag`.
     */
    template<concepts::Tag TTag, typename TQueue, typename T, typename TFn>
    void transform(TQueue& queue, T* out_begin, T* out_end, TFn&& fn, T* in)
    {
        using Idx = typename std::iterator_traits<T*>::difference_type;
        using Acc1D = aV2P::TagToAcc<TTag, aV2P::DimInt<1>, Idx>;

        static_assert(
            std::is_invocable_r_v<T, TFn, T> or std::is_invocable_r_v<T, TFn, Acc1D const&, T>,
            "TFn must accept either one argument (of type T) or two arguments (an accelerator and an argument of type "
            "T), and return a value of type T.");

        Idx size = std::distance(out_begin, out_end);
        detail::TransformKernel<TFn> kernel{fn};

        // Find a valid work division. This could be further optimised.
        auto const config
            = aV2P::KernelCfg<Acc1D>{size, Idx{1}, false, aV2P::GridBlockExtentSubDivRestrictions::Unrestricted};
        auto const grid = aV2P::getValidWorkDiv(config, aV2P::getDev(queue), kernel, in, out_begin, size);

        // Apply the fn function to all elements of the input range.
        aV2P::exec<Acc1D>(queue, grid, kernel, in, out_begin, size);
    }

    /*
     * Applies asynchronously the given function `fn` to the elements of the input buffer `in`,
     * and stores the result in the corresponding elements of the output buffer `out`,
     * using the accelerator back-end identified by `Tag`.
     */
    template<concepts::Tag TTag, typename TQueue, typename TBuf, typename TFn, typename TConstBuf>
    void transform(TQueue& queue, TBuf& out, TFn&& fn, TConstBuf const& in)
    {
        // Check that the input and output buffers have compatible types.
        using Idx = aV2P::Idx<TConstBuf>;
        static_assert(
            std::is_same_v<aV2P::Idx<TBuf>, Idx>,
            "The input and output buffers must have the same index type.");
        using Dim = aV2P::Dim<TConstBuf>;
        static_assert(
            std::is_same_v<aV2P::Dim<TBuf>, Dim>,
            "The input and output buffers must have the same dimension.");
        using In = std::remove_const_t<aV2P::Elem<TConstBuf>>;
        using Out = aV2P::Elem<TBuf>;
        using Vec = aV2P::Vec<Dim, Idx>;
        using Acc = aV2P::TagToAcc<TTag, Dim, Idx>;

        static_assert(
            std::is_invocable_r_v<Out, TFn, In const> or std::is_invocable_r_v<Out, TFn, Acc const&, In const>,
            "TFn must accept either one argument (of the buffer's element type) or two arguments (an accelerator and "
            "the element type), and return a value of the buffer's element type.");

        // Check that the input and output buffers have the same size.
        Vec size = aV2P::getExtents(in);
        assert(aV2P::getExtents(out) == size and "The input and output buffers must have the same extents.");

        // Pass details of the input and output buffers to the kernel:
        //   - address of the first elements
        //   - pitches (in bytes) along all dimensions
        //   - number of elements along all dimensions
        detail::TransformKernelND<TFn> kernel{fn};

        // Find a valid work division. This could be further optimised.
        auto const config = aV2P::KernelCfg<Acc>{
            size,
            Vec::ones(),
            false,
            aV2P::GridBlockExtentSubDivRestrictions::Unrestricted};
        auto const grid = aV2P::getValidWorkDiv(
            config,
            aV2P::getDev(queue),
            kernel,
            in.data(),
            aV2P::getPitchesInBytes(in),
            out.data(),
            aV2P::getPitchesInBytes(out),
            size);

        // Apply the fn function to all elements of the input buffer.
        aV2P::exec<Acc>(
            queue,
            grid,
            kernel,
            in.data(),
            aV2P::getPitchesInBytes(in),
            out.data(),
            aV2P::getPitchesInBytes(out),
            size);
    }

} // namespace aV2P
