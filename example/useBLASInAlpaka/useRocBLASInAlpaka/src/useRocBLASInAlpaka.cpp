/* Copyright 2024  Mehmet Yusufoglu, René Widera, Simeon Ehrig
 * SPDX-License-Identifier: ISC
 */
/*
 * This example uses rocBLAS library functions in alpaka. A rocBLAS function of mutrix multiplication is called by
 * using alpaka buffers and queue. Since the code needs only AccGpuHip backend. Make sure the correct alpaka cmake
 * backend flag is set for alpaka.
 */

#include <aV2P/alpaka.hpp>
#include <rocblas/rocblas.h>

#include <cmath>
#include <iostream>
#include <vector>

// Index type
using Idx = std::size_t;

// Set data type
using DataType = float;

// Initialize the matrix in column-major order (1D buffer)
void initializeMatrix(DataType* buffer, Idx rows, Idx cols)
{
    for(Idx j = 0; j < rows; ++j)
    {
        for(Idx i = 0; i < cols; ++i)
        {
            // Generate some values and set buffer
            buffer[i + j * cols] = static_cast<DataType>((i + j * cols) % 10);
        }
    }
}

auto main() -> int
{
    using Dim1D = aV2P::DimInt<1>;

    // Define matrix dimensions, A is MxK and B is KxN
    Idx const M = 4; // Rows in A and C
    Idx const N = 2; // Columns in B and C
    Idx const K = 3; // Columns in A and rows in B

    // Define the accelerator and queue
    // Use Hip Accelerator. Cmake Acc flags should be set to Hip-Only
    using Acc = aV2P::TagToAcc<aV2P::TagGpuHipRt, Dim1D, Idx>;
    using Queue = aV2P::Queue<Acc, aV2P::Blocking>;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    Queue queue(devAcc);

    // Allocate 1D host memory
    auto bufHostA = aV2P::allocBuf<DataType, Idx>(devHost, M * K);
    auto bufHostB = aV2P::allocBuf<DataType, Idx>(devHost, K * N);
    auto bufHostC = aV2P::allocBuf<DataType, Idx>(devHost, M * N);

    DataType* hostA = aV2P::getPtrNative(bufHostA);
    DataType* hostB = aV2P::getPtrNative(bufHostB);
    DataType* hostC = aV2P::getPtrNative(bufHostC);

    // Initialize host matrices with some values
    initializeMatrix(hostA, M, K);
    initializeMatrix(hostB, K, N);
    std::fill(hostC, hostC + (M * N), 0); // Initialize C with 0s

    // Allocate 1D device memory
    auto bufDevA = aV2P::allocBuf<DataType, Idx>(devAcc, M * K);
    auto bufDevB = aV2P::allocBuf<DataType, Idx>(devAcc, K * N);
    auto bufDevC = aV2P::allocBuf<DataType, Idx>(devAcc, M * N);

    // Copy data to device
    aV2P::memcpy(queue, bufDevA, bufHostA, M * K);
    aV2P::memcpy(queue, bufDevB, bufHostB, K * N);
    aV2P::memcpy(queue, bufDevC, bufHostC, M * N);
    aV2P::wait(queue);

    // Obtain the native HIP stream from the Alpaka queue
    auto rocStream = aV2P::getNativeHandle(queue);
    // rocBLAS setup
    rocblas_handle rocblasHandle;
    rocblas_status status = rocblas_create_handle(&rocblasHandle);

    if(status != rocblas_status_success)
    {
        std::cerr << "rocblas_create_handle failed with status: " << status << std::endl;
        return EXIT_FAILURE;
    }
    // Associate the HIP stream with the rocBLAS handle
    status = rocblas_set_stream(rocblasHandle, rocStream);
    if(status != rocblas_status_success)
    {
        std::cerr << "rocblas_set_stream failed with status: " << status << std::endl;
        rocblas_destroy_handle(rocblasHandle);
        return EXIT_FAILURE;
    }
    // Perform matrix multiplication C = A * B.
    // Set alpha = 1,0f and beta = 0.0f so that the equation C = alpha * A * B + beta * C simplifies to C = A * B.
    float alpha = 1.0f, beta = 0.0f;

    // call general matrix multiply function
    status = rocblas_sgemm(
        rocblasHandle,
        rocblas_operation_none,
        rocblas_operation_none, // No transpose for A and B
        M,
        N,
        K,
        &alpha,
        aV2P::getPtrNative(bufDevA),
        M, // Leading dimension of A
        aV2P::getPtrNative(bufDevB),
        K, // Leading dimension of B
        &beta,
        aV2P::getPtrNative(bufDevC),
        M // Leading dimension of C
    );

    if(status != rocblas_status_success)
    {
        std::cerr << "rocblas_sgemm failed: " << status << std::endl;
        rocblas_destroy_handle(rocblasHandle);
        return EXIT_FAILURE;
    }
    aV2P::wait(queue);

    // Copy result back to host
    aV2P::memcpy(queue, bufHostC, bufDevC, M * N);
    aV2P::wait(queue);

    // Verify the result
    std::cout << "Matrix C (Host):" << std::endl;
    for(Idx j = 0; j < M; ++j)
    {
        for(Idx i = 0; i < N; ++i)
        {
            std::cout << hostC[i + j * N] << " ";
        }
        std::cout << std::endl;
    }

    // Expected values of elements of C
    std::vector<DataType> expectedResult{20, 23, 6, 9, 56, 68, 30, 42};

    // Verify the result
    bool success = true;
    for(Idx j = 0; j < M; ++j)
    {
        for(Idx i = 0; i < N; ++i)
        {
            if(std::fabs(hostC[i + j * N] - expectedResult[i + j * N]) > 1e-5f)
            { // Allow small floating-point errors
                std::cout << "Mismatch at (" << i << ", " << j << "): " << hostC[i + j * N]
                          << " != " << expectedResult[i + j * N] << std::endl;
                success = false;
            }
        }
    }

    std::cout << "Multiplication of matrices of size " << M << "x" << K << " and " << K << "x" << N
              << (success ? " succeeded!" : " failed!") << std::endl;

    if(!success)
    {
        return EXIT_FAILURE;
    }

    // Cleanup
    rocblas_destroy_handle(rocblasHandle);
    return EXIT_SUCCESS;
}
