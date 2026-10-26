# aV2P Migration Guide

aV2P keeps the alpaka 2.2 API surface and executes it on top of a pinned
native alpaka 3.0.0 (commit `eb677e8d7f70d79b35df25d9002f94ba669b6f44`).
Porting an existing 2.2 application is a mechanical rename plus the CMake
replacements below. The only C++ change is device lambdas (Section 3).
See [limitations.md](limitations.md) for what aV2P does not support and why.

## 1. Mechanical substitutions

| 2.2 construct | aV2P construct | Where |
|---------------|----------------|-------|
| namespace `alpaka::` | `aV2P::` | all C++ code |
| `#include <alpaka/` | `#include <aV2P/` | all C++ code |
| `#include <alpaka/alpaka.hpp>` (umbrella) | `#include <aV2P/alpaka.hpp>` | portable code |
| macro `ALPAKA_` | macro `AV2P_` | e.g. `ALPAKA_FN_ACC` becomes `AV2P_FN_ACC` |
| CMake `alpaka_` | CMake `aV2P_` | e.g. `alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE` becomes `aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE` |
| target `alpaka::alpaka` | target `aV2P::aV2P` | `target_link_libraries` |
| `alpaka_add_executable` | `aV2P_add_executable` | CMake |
| `alpaka_add_library` | `aV2P_add_library` | CMake |

No other rename is required. Function names, signatures, template arguments
and behavior stay identical to 2.2.

## 2. CMake integration

### a. add_subdirectory (source tree in the consumer build)

```cmake
cmake_minimum_required(VERSION 3.25)
project(myApp LANGUAGES CXX)

add_subdirectory(${path_to_av2p} aV2P)

aV2P_add_executable(myApp src/main.cpp)
target_link_libraries(myApp PRIVATE aV2P::aV2P)
```

Pass `-DaV2P_ALPAKA3_SOURCE_DIR=/path/to/alpaka3` to use a local alpaka 3
source tree instead of the network fetch.

### b. Installed package

```cmake
cmake_minimum_required(VERSION 3.25)
project(myApp LANGUAGES CXX)

find_package(aV2P CONFIG REQUIRED)

aV2P_add_executable(myApp src/main.cpp)
target_link_libraries(myApp PRIVATE aV2P::aV2P)
```

Configure with `-DCMAKE_PREFIX_PATH=/install/prefix` so both the aV2P and
the alpaka 3 packages are found.

### c. Pinned dependency and reuse variables

aV2P pins alpaka 3.0.0 at the commit above. Reuse variables:

| Variable | Purpose |
|----------|---------|
| `aV2P_ALPAKA3_SOURCE_DIR` | Add an existing alpaka 3 source tree. No network. |
| `aV2P_ALPAKA3_DIR` | Use an installed alpaka 3 package directory. No network. |
| `aV2P_ALPAKA3_OFFLINE` | `ON` forbids any network access. Fails if both of the above are missing. |

Without any of these, `FetchContent` fetches the pinned commit once and
reuses it on reconfigure. aV2P is header-only, requires C++20, and is
version-checked against the pinned native alpaka at configure time.

### Finalization

Use `aV2P_add_executable(target ...)` or `aV2P_add_library(target ...)`.
When CUDA or HIP is enabled these helpers enable the language, tag `.cpp`
sources as CUDA/HIP, and set `CUDA_SEPARABLE_COMPILATION` or `HIP_STANDARD`
on the target. The helpers also schedule `aV2P_finalize(target)` at the end
of the directory scope, so the usual `target_link_libraries(target ...
aV2P::aV2P)` call runs first. You do not call the finalize step yourself.

Native alpaka 3 requires every target that links it to be finalized before
it compiles an alpaka header. `cmake/aV2PFinalize.cmake` defines
`aV2P_finalize(target)` as the aV2P equivalent of `alpaka_finalize()`. A
target created with plain `add_executable` or `add_library` must do the
language and property steps above itself and call
`aV2P_finalize(target)` after `target_link_libraries(target ... aV2P::aV2P)`.
The call is idempotent, accepts a target already finalized by native
`alpaka_finalize()`, and requires the target to link `aV2P::aV2P` or native
alpaka. A target that links `aV2P::aV2P` directly and never finalizes
reports a fatal error at configure time. Targets that reach aV2P through an
interface target are finalized automatically.

Build and run:

```
cmake -S . -B build
cmake --build build --target myApp
./build/myApp
```

## 3. Device lambdas: named functors

Device lambdas are the one documented exception to the mechanical port.
alpaka 3 does not require the legacy extended-lambda mechanism, and the
aV2P adapter does not forward arbitrary device lambda closures. Convert
each `ALPAKA_FN_ACC` lambda to a named functor. Captured values become
members; generic lambdas become a templated `operator()`; the first
parameter stays the accelerator.

Before (2.2, `test/unit/kernel/src/KernelLambda.cpp`):

```cpp
std::uint32_t arg = 42u;
auto kernel = [arg] ALPAKA_FN_ACC(TAcc const& /* acc */, bool* success) -> void
{ ALPAKA_CHECK(*success, 42u == arg); };
REQUIRE(fixture(kernel));
```

After (aV2P, same test):

```cpp
struct KernelWithCapture
{
    std::uint32_t arg;

    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& /* acc */, bool* success) const
    {
        AV2P_CHECK(*success, 42u == arg);
    }
};

KernelWithCapture kernel;
kernel.arg = 42u;
REQUIRE(fixture(kernel));
```

Rules:

- Value-captured variables map to members of the same names.
- `[&]` is not portable. Copy the captures into members explicitly.
- A generic `auto` lambda maps to a functor with a templated `operator()`.
- Stateless lambdas become stateless functors with a single `operator()`.

Every `ALPAKA_FN_ACC` lambda in the 2.2 tree is small. The port for
`example/helloWorldLambda`, `example/reduce` and the kernel, math and
MdSpan unit tests is already in the aV2P branch and uses this pattern.

## 4. Backend selection

Legacy `aV2P_ACC_*_ENABLE` options select backends. The complete option
table with the native alpaka 3 effect and the diagnostic for each value
lives in Section 5 of [limitations.md](limitations.md). Only
`aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE` (serial CPU) has passing test evidence
for end-to-end execution so far; the other backends are tracked in the
compatibility matrix in [limitations.md](limitations.md).

FPGA SYCL support is intentionally removed from aV2P. This is a deliberate
scope decision, not an alpaka 3 capability gap: `aV2P_SYCL_ONEAPI_FPGA` and its
`_MODE`, `_BOARD` and `_BSP` companions, the FPGA types and the FPGA headers
no longer exist, and configuring with a removed option fails with a fatal
error.
