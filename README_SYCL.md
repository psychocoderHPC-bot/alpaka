# Using the SYCL back-end

> :warning: **The SYCL back-end is currently in an experimental state. Bugs are to be expected.** Please report any issue you encounter.

## Supported Compilers

At the moment alpaka's SYCL back-end can only be used together with Intel oneAPI (latest stable version) and supports x86 CPUs and Intel GPUs.

## CMake mode

`icpx` is identified as `IntelLLVM` by CMake. Compilers not based on `icpx` will be detected by CMake which will then print an error message.

### General CMake options

* `alpaka_ACC_SYCL_ENABLE`: set to `ON` to enable the SYCL back-end. Requires the activation of at least one oneAPI hardware target (see below).
* `oneDPL_DIR`: always required. Set to the CMake path of your oneDPL installation. Example: `/opt/intel/oneapi/dpl/2022.1.0/lib/cmake/oneDPL`.

### Building for x86 64-bit CPUs

The following CMake flags can be set for CPUs:

* `alpaka_SYCL_ONEAPI_CPU`: set to `ON` to enable compilation for CPUs. Relies on the Intel OpenCL CPU runtime and ahead-of-time compiler, which support Intel and AMD CPUs.
* `alpaka_SYCL_ONEAPI_CPU_ISA`: the Intel ISA to compile for. Look at the possible `--march` options listed in the output of `opencl-aot --help`. Default: `avx2`.

### Building for Intel GPUs

* `alpaka_SYCL_ONEAPI_GPU`: set to `ON` to enable compilation for Intel GPUs.
* `alpaka_SYCL_ONEAPI_GPU_DEVICES`: semicolon-separated list of one or more Intel GPUs to compile for. The possible values for the devices are listed in the [UsersManual](https://intel.github.io/llvm-docs/UsersManual.html#generic-options) under the flag `-fsycl-targets`. Default: `intel_gpu_pvc`.
  Note: currently only one target at a time can be specified (limitation of the Intel Compiler)

## Standalone mode

### General

Using the SYCL back-end always requires the following flags:

* `-fsycl` (compiler and linker)
* `-fsycl-standard=2020` (compiler)

Device-side printing is possible with `printf`, it calls `sycl::ext::oneapi::experimental::printf` that emulates the standard one. This is an extension of the SYCL standard, still in an experimental state, therefore may not always work correctly.

### Building for x86 64-bit CPUs

1. `#include <alpaka/standalone/CpuSycl.hpp>` in your C++ code.
2. Add the following flags:
  * `-fsycl-targets=spir64_x86_64` (compiler and linker): to enable CPU compilation. Note: If you are using multiple SYCL hardware targets (like CPU and GPU) separate them by comma here.
  * `-Xsycl-target-backend=spir64_x86_64 "-march=<ISA>"` (linker): to choose the Intel ISA to compile for. Check the output of `opencl-aot --help` and look for the possible values of the `--march` flag.

### Building for Intel GPUs

1. `#include <alpaka/standalone/GpuSyclIntel.hpp>` in your C++ code.
2. Add the following flags:
  * `-fsycl-targets=intel_gpu_pvc` (compiler and linker): to enable GPU compilation. Note: If you are using multiple SYCL hardware targets (like CPU and GPU) separate them by comma here.

## Using the SYCL back-end

### Choosing the Accelerator

In contrast to the other back-ends the SYCL back-end comes with multiple different accelerators which should be chosen according to your requirements:

* `alpaka::AccCpuSycl` for targeting Intel and AMD CPUs. In contrast to the other CPU back-ends this will use Intel's OpenCL implementation for CPUs under the hood.
* `alpaka::AccGpuSyclIntel` for targeting Intel GPUs.

These can be used interchangeably (some restrictions apply - see below) with the non-experimental alpaka accelerators to compile an existing alpaka code for SYCL-capable hardware.

### Restrictions

* Similar to the CUDA and HIP back-ends the SYCL back-end only supports up to three kernel dimensions.
* Some Intel GPUs do not support the `double` type for device code. alpaka will not check this.
  You can enable software emulation for `double` precision types with
  ```bash
  export IGC_EnableDPEmulation=1
  export OverrideDefaultFP64Settings=1
  ```
  See [Intel's FAQ](https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-double-precision-emulation-fp64) for more information.
* Managed (shared) memory works but on the GPU it is very slow.
* Managed (shared) memory is incompatible with a user-defined `SIGSEGV` handler. The Intel GPU runtime revokes host
  access to the memory while it is on the device, and installs a `SIGSEGV` handler to migrate the memory back when the
  host accesses it.
  Any other `SIGSEGV` handler installed by the application after the Intel GPU runtime has been initialised will
  intercept these signals, and usually terminate the application.
  For example, Catch2 installs its own POSIX signal handlers at the beginning of each test case. When building the
  alpaka tests for Intel GPUs with the internal Catch2 (`alpaka_USE_INTERNAL_CATCH2=ON`), alpaka disables them by
  setting `CATCH_CONFIG_NO_POSIX_SIGNALS`; an external Catch2 must be built with `-DCATCH_CONFIG_NO_POSIX_SIGNALS=ON`.
  Alternatively, the Intel GPU runtime can be setup to re-install its handler every time it migrates the memory to the
  device:
  ```bash
  export NEOReadDebugKeys=1
  export RegisterPageFaultHandlerOnMigration=1
  ```
  Note that this is an undocumented debug setting of the Intel GPU runtime, and may change without notice.
* The warp operations (`alpaka::warp::activemask`, `all`, `any`, `ballot`, `shfl`, `shfl_up`, `shfl_down` and
  `shfl_xor`) are available only on some SYCL backends and with some oneAPI versions:
  * Intel CPUs and GPUs: starting from oneAPI 2025.2. Up to oneAPI 2026.1 they give wrong results when some threads
    have returned early from the kernel and the SYCL code is compiled without inlining, for example with `-O0` (as in
    CMake's `Debug` builds), `-fno-inline` or `-fno-inline-functions`: the threads that have returned are still
    considered active. In these builds the warp operations are not available, unless
    `ALPAKA_SYCL_DISABLE_WARP_INLINE_CHECK` is defined, for example if the kernels using them never return early.
  * NVIDIA GPUs: starting from oneAPI 2025.0, also in `Debug` builds.
  * AMD GPUs: not available.

  Using a warp operation where it is not available fails the compilation with a `static_assert` that explains why.
  `alpaka::warp::getSize` is always available.
* With oneAPI 2025.0 the NVIDIA GPU backend does not support `Debug` builds, because `ptxas` fails on the device debug
  information: CMake stops with an error.
* The latest Intel OpenCL CPU runtime does not work properly. Some tests (`atomicTest`, `blockSharedTest`, `blockSharedSharingTest` and `warpTest`) fail with a `PI_ERROR_OUT_OF_RESOURCES`. The only runtime version that seems to work is 2022.14.8.0.04 (can be downloaded [here](https://github.com/intel/llvm/releases/download/2022-WW33/oclcpuexp-2022.14.8.0.04_rel.tar.gz)) apart from a bug with `all_of_group` / `any_of_group` that requires the warp size being equal to the block size as a workaround.

### Choosing the sub-group size (warp size)

Most SYCL targets support multiple sub-group sizes. There is a trait to specify at compile time the sub-group size to use for a kernel. For example, if `MyKernel` requires a sub-group size of 32, this can be declared specialising the `alpaka::trait::WarpSize`:
```cpp
struct MyKernel { ... };

template<typename TAcc>
struct alpaka::trait::WarpSize<MyKernel, TAcc>
    : std::integral_constant<std::uint32_t, 32>
{
};
```
This can be extended to kernels that support multiple sub-group sizes at compile time:
```cpp
template<std::uint32_t TWarpSize>
struct MyKernel { ... };

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<MyKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};
```

The default behaviour, when no sub-group size is specified, is to let the back-end compiler pick the preferred size.

Before launching a kernel with a compile-time sub-group size the user should query the sizes supported by the device, and choose accordingly. If the device does not support the requested size, the SYCL runtime will throw a synchronous exception.

During just-in-time (JIT) compilation this guarantees that a kernel is compiled only for the sizes supported by the device. During ahead-of-time (AOT) compilation this is not enough, because the device is not known at compile time. The SYCL specification mandates that the back-end compilers should not fail if a kernel uses unsupported features, like unsupported sub-group sizes. Unfortunately the Intel OpenCL CPU and GPU compilers currently fail with a hard error. To work around this limitation, use the preprocessor macros defined when compiling AOT for the new SYCL targets to enable the compilation only for the sub-group sizes supported by each device.

Note: while the CPU OpenCL back-end supports a sub-group size of 64, Intel's SYCL implementation currently does not. To avoid issues with the sub-group primitives, alpaka always considers the sub-group size of 64 as not supported by the device.
