/* Copyright 2025 Axel Hübl, Benjamin Worpitz, Matthias Werner, Sergei Bastrakov, René Widera, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Andrea Bocci, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/atomic/Traits.hpp>
#include <aV2P/math/FloatEqualExact.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <climits>
#include <type_traits>

template<typename T1, typename T2>
AV2P_FN_INLINE AV2P_FN_HOST_ACC auto equals(T1 a, T2 b) -> bool
{
    return a == b;
}

AV2P_FN_INLINE AV2P_FN_HOST_ACC auto equals(float a, float b) -> bool
{
    return aV2P::math::floatEqualExactNoWarning(a, b);
}

AV2P_FN_INLINE AV2P_FN_HOST_ACC auto equals(double a, double b) -> bool
{
    return aV2P::math::floatEqualExactNoWarning(a, b);
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicAdd(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = static_cast<T>(operandOrig + value);
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicAdd>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicAdd(acc, &operand, value, aV2P::hierarchy::Threads{});
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicSub(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = static_cast<T>(operandOrig - value);
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicSub>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicSub(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicMin(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = (operandOrig < value) ? operandOrig : value;
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicMin>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicMin(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicMax(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = (operandOrig > value) ? operandOrig : value;
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicMax>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicMax(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicExch(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = value;
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicExch>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicExch(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicInc(TAcc const& acc, bool* success, T operandOrig) -> void
{
    // \TODO: Check reset to 0 at 'value'.
    T const value = static_cast<T>(42);
    T const reference = static_cast<T>(operandOrig + 1);
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicInc>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicInc(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicInc(acc, &operand);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicDec(TAcc const& acc, bool* success, T operandOrig) -> void
{
    // \TODO: Check reset to 'value' at 0.
    T const value = static_cast<T>(42);
    T const reference = static_cast<T>(operandOrig - 1);
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicDec>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicDec(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicDec(acc, &operand);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicAnd(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = operandOrig & value;
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicAnd>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicAnd(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicOr(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = operandOrig | value;
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicOr>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOr(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicXor(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(operandOrig + static_cast<T>(4));
    T const reference = operandOrig ^ value;
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = aV2P::atomicOp<aV2P::AtomicXor>(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = aV2P::atomicXor(acc, &operand, value);
        AV2P_CHECK(*success, equals(operandOrig, ret));
        AV2P_CHECK(*success, equals(operand, reference));
    }
}

AV2P_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
AV2P_FN_ACC auto testAtomicCas(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    auto& operand = aV2P::declareSharedVar<T, __COUNTER__>(acc);

    // with match
    {
        T const compare = operandOrig;
        T const reference = value;
        {
            operand = operandOrig;
            T const ret = aV2P::atomicOp<aV2P::AtomicCas>(acc, &operand, compare, value);
            AV2P_CHECK(*success, equals(operandOrig, ret));
            AV2P_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operandOrig;
            T const ret = aV2P::atomicCas(acc, &operand, compare, value);
            AV2P_CHECK(*success, equals(operandOrig, ret));
            AV2P_CHECK(*success, equals(operand, reference));
        }
    }

    // without match
    {
        T const compare = static_cast<T>(operandOrig + static_cast<T>(1));
        T const reference = operandOrig;
        {
            operand = operandOrig;
            T const ret = aV2P::atomicOp<aV2P::AtomicCas>(acc, &operand, compare, value);
            AV2P_CHECK(*success, equals(operandOrig, ret));
            AV2P_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operandOrig;
            T const ret = aV2P::atomicCas(acc, &operand, compare, value);
            AV2P_CHECK(*success, equals(operandOrig, ret));
            AV2P_CHECK(*success, equals(operand, reference));
        }
    }
}

template<typename TAcc, typename T, typename Sfinae = void>
class AtomicTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TFacade>
    AV2P_FN_ACC auto operator()(TFacade const& acc, bool* success, T operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        if constexpr(std::is_unsigned_v<T>)
        {
            // atomicInc / atomicDec are implemented only for unsigned integer types
            testAtomicInc(acc, success, operandOrig);
            testAtomicDec(acc, success, operandOrig);
        }

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TAcc, typename T>
class AtomicTestKernel<TAcc, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TFacade>
    AV2P_FN_ACC auto operator()(TFacade const& acc, bool* success, T operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // These are not supported on float/double types
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

// The narrow-type stub is specialized on the concrete accelerator alias of the enabled GPU backend.
// The legacy three-parameter `AccGpuUniformCudaHipRt` name expands to an alias that drops the legacy
// API tag, which clang rejects as a deduction pattern in a partial specialization.
#if defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA

template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    aV2P::AccGpuCudaRt<TDim, TIdx>,
    T,
    std::enable_if_t<sizeof(T) != 4u && sizeof(T) != 8u>>
{
public:
    AV2P_NO_HOST_ACC_WARNING
    AV2P_FN_ACC auto operator()(
        aV2P::AccGpuCudaRt<TDim, TIdx> const& /* acc */,
        bool* success,
        T /* operandOrig */) const -> void
    {
        // All other types are not supported by CUDA/HIP atomic operations.
        AV2P_CHECK(*success, true);
    }
};

#elif defined(AV2P_ACC_GPU_HIP_ENABLED) && AV2P_LANG_HIP

template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    aV2P::AccGpuHipRt<TDim, TIdx>,
    T,
    std::enable_if_t<sizeof(T) != 4u && sizeof(T) != 8u>>
{
public:
    AV2P_NO_HOST_ACC_WARNING
    AV2P_FN_ACC auto operator()(
        aV2P::AccGpuHipRt<TDim, TIdx> const& /* acc */,
        bool* success,
        T /* operandOrig */) const -> void
    {
        // All other types are not supported by CUDA/HIP atomic operations.
        AV2P_CHECK(*success, true);
    }
};

#endif

template<typename TAcc, typename T>
struct TestAtomicOperations
{
    static auto testAtomicOperations() -> void
    {
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;

        aV2P::test::KernelExecutionFixture<TAcc> fixture(aV2P::Vec<Dim, Idx>::ones());

        AtomicTestKernel<TAcc, T> kernel;

        T value = static_cast<T>(32);
        REQUIRE(fixture(kernel, value));
    }
};

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("atomicOperationsWorking", "[atomic]", TestAccs)
{
    using Acc = TestType;

    // According to the CUDA 12.1 Programming Guide, Section 7.14. Atomic Functions, an atomic function performs a
    // read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory.
    // Some operations require a compute capability of 5.0, 6.0, or higher; on older devices they can be emulated with
    // an atomicCAS loop.

    // According to SYCL 2020 rev. 7, Section 4.15.3. Atomic references, the template parameter T must be one of the
    // following types:
    //   - int, unsigned int,
    //   - long, unsigned long,
    //   - long long, unsigned long long,
    //   - float, or double.
    // In addition, the type T must satisfy one of the following conditions:
    //  - sizeof(T) == 4, or
    //  - sizeof(T) == 8 and the code containing the atomic_ref was submitted to a device that has aspect::atomic64.

    TestAtomicOperations<Acc, unsigned int>::testAtomicOperations();
    TestAtomicOperations<Acc, int>::testAtomicOperations();

    // Intel FPGA SYCL backend does not support 64-bit atomics
    if constexpr(sizeof(long) == 4 or not std::is_same_v<aV2P::AccToTag<Acc>, aV2P::TagFpgaSyclIntel>)
    {
        TestAtomicOperations<Acc, unsigned long>::testAtomicOperations();
        TestAtomicOperations<Acc, long>::testAtomicOperations();
    }

    // Intel FPGA SYCL backend does not support 64-bit atomics
    if constexpr(not std::is_same_v<aV2P::AccToTag<Acc>, aV2P::TagFpgaSyclIntel>)
    {
        TestAtomicOperations<Acc, unsigned long long>::testAtomicOperations();
        TestAtomicOperations<Acc, long long>::testAtomicOperations();
    }

    TestAtomicOperations<Acc, float>::testAtomicOperations();

    // Intel FPGA SYCL backend does not support 64-bit atomics
    if constexpr(not std::is_same_v<aV2P::AccToTag<Acc>, aV2P::TagFpgaSyclIntel>)
    {
        TestAtomicOperations<Acc, double>::testAtomicOperations();
    }
}
