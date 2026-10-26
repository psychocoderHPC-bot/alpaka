/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "aV2P/exec/UniformElements.hpp"

#include "WorkDiv.hpp"
#include "aV2P/acc/Traits.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/exec/Once.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/platform/PlatformCpu.hpp"
#include "aV2P/platform/Traits.hpp"
#include "aV2P/queue/Traits.hpp"
#include "aV2P/test/acc/TestAccs.hpp"
#include "aV2P/wait/Traits.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <type_traits>

#if AV2P_COMP_MSVC
// MSVC uses __restrict instead of __restrict__.
#    define __restrict__ __restrict
#endif

#if AV2P_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
// Global Host object used by all tests.
using Host = aV2P::DevCpu;
static Host host = aV2P::getDevByIdx(aV2P::PlatformCpu{}, 0);
#if AV2P_COMP_CLANG
#    pragma clang diagnostic pop
#endif

struct VectorAddKernel
{
    template<typename TAcc, typename T>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Idx<TAcc> size) const
    {
        for(auto index : aV2P::uniformElements(acc, size))
        {
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernelSkip
{
    template<typename TAcc, typename T>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Idx<TAcc> first,
        aV2P::Idx<TAcc> size) const
    {
        for(auto index : aV2P::uniformElements(acc, first, size))
        {
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel1D
{
    template<typename TAcc, typename T, typename = std::enable_if_t<aV2P::Dim<TAcc>::value == 1u>>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> size) const
    {
        for(auto ndindex : aV2P::uniformElementsND(acc, size))
        {
            auto index = ndindex[0];
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel2D
{
    template<typename TAcc, typename T, typename = std::enable_if_t<aV2P::Dim<TAcc>::value == 2u>>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> size) const
    {
        for(auto ndindex : aV2P::uniformElementsND(acc, size))
        {
            auto index = ndindex[0] * size[1] + ndindex[1];
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel3D
{
    template<typename TAcc, typename T, typename = std::enable_if_t<aV2P::Dim<TAcc>::value == 3u>>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> size) const
    {
        for(auto ndindex : aV2P::uniformElementsND(acc, size))
        {
            auto index = (ndindex[0] * size[1] + ndindex[1]) * size[2] + ndindex[2];
            out[index] = in1[index] + in2[index];
        }
    }
};

/* This is not an efficient approach, and it uses more operations and synchronisations than needed. It is written like
 * this to test the use of dynamic shared memory, split block and element loops, and block-level synchronisations.
 */

struct VectorAddBlockKernel
{
    template<typename TAcc, typename T>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Idx<TAcc> size) const
    {
        using Idx = aV2P::Idx<TAcc>;

        // Get the block size.
        auto const blockSize = aV2P::getWorkDiv<aV2P::Block, aV2P::Elems>(acc)[0u];

        // Get the dynamic shared memory buffer.
        T* buffer = aV2P::getDynSharedMem<T>(acc);

        // Split the loop over the elements into an outer loop over the groups and an inner loop over the elements, to
        // permit the synchronisation of the threads in each block between each step: the outer loop is needed to
        // repeat the "block" as many times as needed to cover the whole problem space; the inner loop is needed for
        // work division with more than one element per thread.
        for(auto block : aV2P::uniformGroups(acc, size))
        {
            // Initialise the shared memory.
            if(aV2P::oncePerBlock(acc))
            {
                for(Idx local = 0; local < blockSize; ++local)
                {
                    buffer[local] = 0;
                }
            }
            // Synchronise all threads in the block.
            aV2P::syncBlockThreads(acc);
            // Accumulate the first set of data into shared memory.
            for(auto index : aV2P::uniformGroupElements(acc, block, size))
            {
                buffer[index.local] += in1[index.global];
            }
            // Synchronise all threads in the block.
            aV2P::syncBlockThreads(acc);
            // Accumulate the second set of data into shared memory.
            for(auto index : aV2P::uniformGroupElements(acc, block, size))
            {
                buffer[index.local] += in2[index.global];
            }
            // Synchronise all threads in the block.
            aV2P::syncBlockThreads(acc);
            // Store the results into global memory.
            for(auto index : aV2P::uniformGroupElements(acc, block, size))
            {
                out[index.global] = buffer[index.local];
            }
            // Synchronise all threads in the block; this is necessary to avoid race conditions between different
            // iterations of the uniformGroups loop.
            aV2P::syncBlockThreads(acc);
        }
    }
};

/* Run all operations in a single thread.
 * Written in an inefficient way to test "oncePerGrid".
 */

struct VectorAddKernelSerial
{
    template<typename TAcc, typename T>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Idx<TAcc> size) const
    {
        using Idx = aV2P::Idx<TAcc>;

        // The operations are performed by a single thread.
        if(aV2P::oncePerGrid(acc))
        {
            for(Idx index = 0; index < size; ++index)
            {
                // Unsafe, used to test that each element is summed exactly once.
                out[index] += in1[index];
                out[index] += in2[index];
            }
        }
    }
};

/* Run all operations in one thread per block.
 * Written in an inefficient way to test "oncePerBlock".
 */

struct VectorAddKernelBlockSerial
{
    template<typename TAcc, typename T>
    AV2P_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        aV2P::Idx<TAcc> size) const
    {
        using Idx = aV2P::Idx<TAcc>;

        // Get the block size.
        auto const blockSize = aV2P::getWorkDiv<aV2P::Block, aV2P::Elems>(acc)[0u];
        // The loop is used to repeat the "block" as many times as needed to cover the whole problem space.
        for(auto block : aV2P::uniformGroups(acc, size))
        {
            // The operations are performed by a single thread in each "logical" block.
            auto const first = blockSize * block;
            auto const range = std::min<Idx>(first + blockSize, size);
            if(aV2P::oncePerBlock(acc))
            {
                for(Idx index = first; index < range; ++index)
                {
                    // Unsafe, used to test that each element is summed exactly once.
                    out[index] += in1[index];
                    out[index] += in2[index];
                }
            }
        }
    }
};

namespace aV2P::trait
{
    // Specialize the BlockSharedMemDynSizeBytes trait to specify the amount of block shared dynamic memory for the
    // VectorAddBlockKernel kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<VectorAddBlockKernel, TAcc>
    {
        using Idx = aV2P::Idx<TAcc>;
        using Dim1D = aV2P::DimInt<1u>;
        using Vec1D = aV2P::Vec<Dim1D, Idx>;

        // The size in bytes of the shared memory allocated for a block.
        template<typename T>
        AV2P_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
            VectorAddBlockKernel const& /* kernel */,
            Vec1D threads,
            Vec1D elements,
            T const* __restrict__ /* in1 */,
            T const* __restrict__ /* in2 */,
            T* __restrict__ /* out */,
            Idx /* size */)
        {
#if defined(__GNUC__)
            // Silence a potential warning about
            // warning: conversion to ‘long unsigned int’ from ‘long int’ may change the sign of the result
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
            return static_cast<std::size_t>(threads[0] * elements[0] * sizeof(T));
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
        }
    };
} // namespace aV2P::trait

// Test the 1-dimensional kernel on all devices.
template<typename TAcc, typename TKernel>
void testVectorAddKernel(
    aV2P::Idx<TAcc> problem_size,
    aV2P::Idx<TAcc> grid_size,
    aV2P::Idx<TAcc> block_size,
    TKernel kernel)
{
    using Acc = TAcc;
    using Idx = aV2P::Idx<Acc>;
    using Platform = aV2P::Platform<Acc>;
    using Device = aV2P::Dev<Platform>;
    using Queue = aV2P::Queue<Device, aV2P::NonBlocking>;

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // Tolerance.
    constexpr float epsilon = 0.000001f;

    // Buffer size.
    const Idx size = problem_size;

    // Initialise the accelerator platform.
    Platform platform{};

    // Allocate input and output host buffers in pinned memory accessible by the Platform devices.
    auto in1_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto in2_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto out_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffers with random data, and the output buffer with zeros.
    for(Idx i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.f;
    }

    // Run the test on each device.
    for(auto const& device : aV2P::getDevs(platform))
    {
        /* clang-format off */
        INFO("Test 1D vector addition on " << aV2P::getName(device) << " over " << problem_size << " values with "
                                           << grid_size << " blocks of " << block_size << " elements");
        /* clang-format on */
        auto queue = Queue(device);

        // Allocate input and output buffers on the device.
        auto in1_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto in2_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto out_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);

        // Copy the input data to the device; the size is known from the buffer objects.
        aV2P::memcpy(queue, in1_d, in1_h);
        aV2P::memcpy(queue, in2_d, in2_h);

        // Fill the output buffer with zeros; the size is known from the buffer objects.
        aV2P::memset(queue, out_d, 0);

        // Launch the 1-dimensional kernel with scalar size.
        auto div = makeWorkDiv<Acc>(grid_size, block_size);
        aV2P::exec<Acc>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), size);

        // Copy the results from the device to the host.
        aV2P::memcpy(queue, out_h, out_d);

        // Wait for all the operations to complete.
        aV2P::wait(queue);

        // Check the results.
        for(Idx i = 0; i < size; ++i)
        {
            float sum = in1_h[i] + in2_h[i];
            REQUIRE(out_h[i] < sum + epsilon);
            REQUIRE(out_h[i] > sum - epsilon);
        }
    }
}

// Test the 1-dimensional kernel on all devices, potentially skipping some elements.
template<typename TAcc, typename TKernel>
void testVectorAddKernelSkip(
    aV2P::Idx<TAcc> skip_elements,
    aV2P::Idx<TAcc> problem_size,
    aV2P::Idx<TAcc> grid_size,
    aV2P::Idx<TAcc> block_size,
    TKernel kernel)
{
    using Acc = TAcc;
    using Idx = aV2P::Idx<Acc>;
    using Platform = aV2P::Platform<Acc>;
    using Device = aV2P::Dev<Platform>;
    using Queue = aV2P::Queue<Device, aV2P::NonBlocking>;

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // Tolerance.
    constexpr float epsilon = 0.000001f;

    // Buffer size.
    const Idx size = problem_size;

    // Initialise the accelerator platform.
    Platform platform{};

    // Allocate input and output host buffers in pinned memory accessible by the Platform devices.
    auto in1_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto in2_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto out_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffers with random data, and the output buffer with zeros.
    for(Idx i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.f;
    }

    // Run the test on each device.
    for(auto const& device : aV2P::getDevs(platform))
    {
        /* clang-format off */
        INFO("Test 1D vector addition on " << aV2P::getName(device) << " skipping " << skip_elements << " over "
                                           << problem_size << " values with " << grid_size << " blocks of "
                                           << block_size << " elements");
        /* clang-format on */
        auto queue = Queue(device);

        // Allocate input and output buffers on the device.
        auto in1_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto in2_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto out_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);

        // Copy the input data to the device; the size is known from the buffer objects.
        aV2P::memcpy(queue, in1_d, in1_h);
        aV2P::memcpy(queue, in2_d, in2_h);

        // Fill the output buffer with zeros; the size is known from the buffer objects.
        aV2P::memset(queue, out_d, 0);

        // Launch the 1-dimensional kernel with scalar size.
        auto div = makeWorkDiv<Acc>(grid_size, block_size);
        aV2P::exec<Acc>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), skip_elements, size);

        // Copy the results from the device to the host.
        aV2P::memcpy(queue, out_h, out_d);

        // Wait for all the operations to complete.
        aV2P::wait(queue);

        // Check the results.
        for(Idx i = 0; i < skip_elements; ++i)
        {
            // The first part of the output vector should not have been modified at all, and should be identically 0.
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
            REQUIRE(out_h[i] == 0);
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
        }
        for(Idx i = skip_elements; i < size; ++i)
        {
            float sum = in1_h[i] + in2_h[i];
            REQUIRE(out_h[i] < sum + epsilon);
            REQUIRE(out_h[i] > sum - epsilon);
        }
    }
}

// Test the N-dimensional kernels on all devices.
template<typename TAcc, typename TKernel>
void testVectorAddKernelND(
    aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> problem_size,
    aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> grid_size,
    aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> block_size,
    TKernel kernel)
{
    using Acc = TAcc;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Platform = aV2P::Platform<Acc>;
    using Device = aV2P::Dev<Platform>;
    using Queue = aV2P::Queue<Device, aV2P::NonBlocking>;

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // Tolerance.
    constexpr float epsilon = 0.000001f;

    // Linearised buffer size.
    const Idx size = problem_size.prod();

    // Initialise the accelerator platform.
    Platform platform{};

    // Allocate input and output host buffers in pinned memory accessible by the Platform devices.
    auto in1_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto in2_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);
    auto out_h = aV2P::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffers with random data, and the output buffer with zeros.
    for(Idx i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.f;
    }

    // Run the test on each device.
    for(auto const& device : aV2P::getDevs(platform))
    {
        /* clang-format off */
        INFO("Test " << Dim::value << "D vector addition on " << aV2P::getName(device) << " over " << problem_size
                     << " values with " << grid_size << " blocks of " << block_size << " elements");
        /* clang-format on */
        auto queue = Queue(device);

        // Allocate input and output buffers on the device.
        auto in1_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto in2_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto out_d = aV2P::allocAsyncBufIfSupported<float, Idx>(queue, size);

        // Copy the input data to the device; the size is known from the buffer objects.
        aV2P::memcpy(queue, in1_d, in1_h);
        aV2P::memcpy(queue, in2_d, in2_h);

        // Fill the output buffer with zeros; the size is known from the buffer objects.
        aV2P::memset(queue, out_d, 0);

        // Launch the 3-dimensional kernel.
        auto div = makeWorkDiv<Acc>(grid_size, block_size);
        aV2P::exec<Acc>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), problem_size);

        // Copy the results from the device to the host.
        aV2P::memcpy(queue, out_h, out_d);

        // Wait for all the operations to complete.
        aV2P::wait(queue);

        // Check the results.
        for(Idx i = 0; i < size; ++i)
        {
            float sum = in1_h[i] + in2_h[i];
            REQUIRE(out_h[i] < sum + epsilon);
            REQUIRE(out_h[i] > sum - epsilon);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("UniformElements", "[exec]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Vec = aV2P::Vec<Dim, Idx>;

    // 1-dimensional kernels.
    if constexpr(Dim::value == 1)
    {
        SECTION("VectorAddKernel1D, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector addition with small block size");
            testVectorAddKernelND<TestType, VectorAddKernel1D>(Vec{10000}, Vec{32}, Vec{32}, VectorAddKernel1D{});
        }

        SECTION("VectorAddKernel1D, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector addition with large block size");
            testVectorAddKernelND<TestType>({10}, {2}, {32}, VectorAddKernel1D{});
        }
    }

    // 2-dimensional kernels.
    if constexpr(Dim::value == 2)
    {
        SECTION("VectorAddKernel2D, small block size")
        {
            // Launch the 2-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 2D vector addition with small block size");
            testVectorAddKernelND<TestType>({400, 250}, {5, 4}, {8, 4}, VectorAddKernel2D{});
        }

        SECTION("VectorAddKernel2D, large block size")
        {
            // Launch the 2-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 2D vector addition with large block size");
            testVectorAddKernelND<TestType>({5, 3}, {2, 2}, {8, 4}, VectorAddKernel2D{});
        }
    }

    // 3-dimensional kernels.
    if constexpr(Dim::value == 3)
    {
        SECTION("VectorAddKernel3D, small block size")
        {
            // Launch the 3-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 3D vector addition with small block size");
            testVectorAddKernelND<TestType>({50, 25, 16}, {5, 2, 2}, {2, 4, 4}, VectorAddKernel3D{});
        }

        SECTION("VectorAddKernel3D, large block size")
        {
            // Launch the 3-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 3D vector addition with large block size");
            testVectorAddKernelND<TestType>({2, 3, 3}, {2, 2, 2}, {2, 4, 4}, VectorAddKernel3D{});
        }
    }

    // 1-dimensional kernels.
    if constexpr(Dim::value == 1)
    {
        SECTION("VectorAddKernel, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddKernel{});
        }

        SECTION("VectorAddKernel, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddKernel{});
        }

        SECTION("VectorAddBlockKernel, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector block-level addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddBlockKernel{});
        }

        SECTION("VectorAddBlockKernel, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector block-level addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddBlockKernel{});
        }

        SECTION("VectorAddKernelSerial, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector single-threaded serial addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddKernelSerial{});
        }

        SECTION("VectorAddKernelSerial, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector single-threaded seria addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddKernelSerial{});
        }

        SECTION("VectorAddKernelBlockSerial, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector block-level serial addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddKernelBlockSerial{});
        }

        SECTION("VectorAddKernelBlockSerial, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector block-level serial addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddKernelBlockSerial{});
        }

        SECTION("VectorAddKernelSkip, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector addition with small block size, using scalar dimensions");
            testVectorAddKernelSkip<TestType>(20, 10000, 32, 32, VectorAddKernelSkip{});
        }

        SECTION("VectorAddKernelSkip, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector addition with large block size, using scalar dimensions");
            testVectorAddKernelSkip<TestType>(2, 10, 2, 32, VectorAddKernelSkip{});
        }
    }
}
