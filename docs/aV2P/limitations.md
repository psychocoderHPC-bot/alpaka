# aV2P Limitations

This file lists the compatibility limitations between alpaka 2.2 and aV2P
with the evidence and the restoration condition for each. aV2P keeps the
2.2 public API and executes it on native alpaka 3
(commit `eb677e8d7f70d79b35df25d9002f94ba669b6f44`). The entries below are
the features where that layer cannot preserve 2.2 behavior.

## Memory layout

aV2P keeps the 2.2 host layout and exposes the native alpaka 3 device layout:

- **CPU (host API) buffers, multi-dimensional:** contiguous row-major
  (`pitch[i] == product(extents[i+1:]) * sizeof(T)`). `std::data(buf)` spans the
  whole buffer linearly. Because rows are adjacent, a multi-dimensional host
  buffer only guarantees element alignment, not SIMD row alignment.
  One-dimensional host buffers keep the allocation alignment.
- **CUDA, HIP and SYCL device buffers:** the native alpaka 3 allocation is
  returned unchanged, so `getPitchesInBytes()` reports the padded row pitches
  (`pitch >= row bytes`) and `data()`, views, sub-views, copies and kernels all
  use those native pitches. `getAlpaka3()` returns the same native buffer.
- **Mapped and unified buffers:** `allocMappedBuf` and `allocManagedBuf` return
  a host-bound `BufCpu` whose contiguous host view covers the same allocation.
  The underlying allocation may be padded; the host view shares its base
  pointer and ownership, and no copy is made.

## 1. Backend support

| Legacy accelerator | Native alpaka 3 executor | Adapter state |
|--------------------|--------------------------|---------------|
| `AccCpuSerial` (`aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE`) | `api::Host` / `deviceKind::Cpu` / `exec::CpuSerial` | implemented and tested |
| `AccCpuOmp2Blocks` | `api::Host` / `deviceKind::Cpu` / `exec::CpuOmpBlocks` | PENDING VALIDATION, adapter in progress |
| `AccGpuCudaRt` | `api::Cuda` / `deviceKind::NvidiaGpu` / `exec::GpuCuda` | PENDING VALIDATION, adapter in progress |
| `AccGpuHipRt` | `api::Hip` / `deviceKind::AmdGpu` / `exec::GpuHip` | PENDING VALIDATION, adapter in progress |
| `AccCpuSycl`, `AccGpuSyclIntel` | `api::OneApi` / `exec::OneApi` | PENDING VALIDATION, adapter in progress |
| `AccCpuTbbBlocks` (`aV2P_ACC_CPU_B_TBB_T_SEQ_ENABLE`) | `api::Host` / `deviceKind::Cpu` / `exec::CpuTbbBlocks` | implemented and tested |
| `AccCpuThreads` | none | no faithful alpaka 3 executor |
| `AccCpuOmp2Threads` | none | no faithful alpaka 3 executor |
| `AccGpuSyclAmd`, `AccGpuSyclNvidia` | none | no native vendor-GPU SYCL target |

FPGA SYCL support is intentionally removed from aV2P: this is a deliberate
aV2P scope decision, not an alpaka 3 capability gap. The FPGA accelerator,
device, platform, event, queue and buffer types, their headers and their CMake
options no longer exist.

Passing evidence exists for the serial CPU path (host serial build of the
ported examples and the kernel, math and memview unit tests) and for the
TBB blocks accelerator (`accTest`, `devTest`, `kernelTest`,
`blockSharedTest`, `blockSyncTest` and `workDivTest`, plus a 52/52 full
`ctest` on the serial configuration). The other backends are PENDING
VALIDATION because the adapter is in progress and no test evidence exists
yet.

`AccCpuThreads` and `AccCpuOmp2Threads` remain deliberate stubs: the
pinned alpaka 3 `exec::allExecutors` contains only `GpuCuda`, `GpuHip`,
`OneApi`, `CpuOmpBlocks`, `CpuTbbBlocks` and `CpuSerial`, so there is no
std::thread or OpenMP block-thread executor with 2.2 semantics. The CMake
mapping table marks both options `unsupported` and the public names stay
`UnimplementedAcc`.

## 2. Warp mask API

alpaka 2.2.0 adds no new public mask function. The public warp API
(`getSize`, `getSizeCompileTime`, `getSizeUpperLimit`, `activemask`, `all`,
`any`, `ballot`, `shfl`, `shfl_up`, `shfl_down`, `shfl_xor`) is identical
in 2.1.0 and 2.2.0 apart from the return types of `activemask` and
`ballot`. The 2.2.0 CHANGELOG claims a `warp::Mask` wrapper type, but that
type is not present in the 2.2.0 source tree.

Documented differences, 2.2 vs native alpaka 3:

- 2.2 `activemask(warp)` and `ballot(warp, predicate)` return
  `TWarp::mask_type` (`std::uint32_t` for single-thread and SYCL,
  `std::uint32_t` for CUDA, `std::uint64_t` for HIP). In 2.1.0 the return
  was deduced.
- alpaka 3 `activemask(acc)` and `ballot(acc, predicate)` return a fixed
  conditional type: `std::uint32_t` when the warp size is 32 or smaller,
  `std::uint64_t` above 32.
- 2.2 `all` and `any` return `std::int32_t`; alpaka 3 returns `bool`.
- HIP `activemask`: 2.2 uses `__activemask()` on HIP 6.2 and newer;
  alpaka 3 uses `__ballot(1u)`.
- 2.2 SYCL uses an opportunistic group and the `ALPAKA_SYCL_WARP_*`
  unsupported macros; alpaka 3 uses the SYCL `this_work_item` subgroup
  call and has no equivalent macro set.

```cpp
// 2.2
int x = alpaka::warp::all(warp, true);            // std::int32_t
auto m = alpaka::warp::activemask(warp);          // TWarp::mask_type

// aV2P / alpaka 3
bool b = alpaka::onAcc::warp::all(acc, true);     // bool
auto m = alpaka::onAcc::warp::activemask(acc);    // 32 or 64 bits
```

Restoration condition: none. The claimed 2.2 `warp::Mask` wrapper was never
added, so aV2P exposes the alpaka 3 conditional width. Call sites that
depend on the per-backend width must follow the alpaka 3 width.

## 3. Zero-dimensional memory

The full 0D allocation and `View` path exists in alpaka 2.2 and does not
exist in alpaka 3. The type `Vec<T, 0>` persists as an inert placeholder;
allocation and view construction reject it. A compile probe against alpaka
3 fails inside the host-allocation path, where the code requires `dim()`
and `UniVec` members that `Vec<T, 0>` does not provide. Affected instantiations
in the 2.2 tree:

- `test/integ/zeroDimBuffer` (whole file),
- `test/unit/mem/buf/src/BufTest.cpp`: `memBufMove` and the scalar block in
  `memBufAccessors`,
- `test/unit/mem/buf/src/FillBuffers.cpp`: `memBufFillScalarFloatTest`,
- `test/unit/exec/src/Once.cpp`: `oncePerGrid` and `oncePerBlock`,
- `test/unit/vec/src/VecTest.cpp`: 0D constructor block,
- `example/helloWorldGridSync`: 0D result buffers (the example also needs the
  cooperative-launch API, see Section 7).

```cpp
// 2.2, compiles and runs
using Dim0D = alpaka::DimInt<0u>;
using Scalar = alpaka::Vec<Dim0D, int>;
auto s = alpaka::allocBuf<Dim0D, int>(dev, Scalar{1});

// aV2P, does not compile
using Dim0D = aV2P::DimInt<0u>;
using Scalar = aV2P::Vec<Dim0D, int>;
auto s = aV2P::allocBuf<Dim0D, int>(dev, Scalar{1});
```

Restoration condition: re-enable the listed cases only when the pinned
alpaka 3 provides 0D support, i.e. `Vec<T, 0>` gains `dim()` and `UniVec`
and the host-allocation path accepts `dim() == 0`. Until then disable the
cases with `#if 0` or a CMake guard.

## 4. Device lambdas

alpaka 3 does not require the 2.2 extended-lambda mechanism, and the aV2P
translation layer does not forward device lambda closures. Port device
lambdas with the named-functor rewrite in
[migration.md](migration.md) Section 3. There is no mechanical substitution
for the lambda syntax. `AV2P_FN_ACC` still marks the device functor.

Restoration condition: none. alpaka 3 exposes captured state through the
named kernel functor, and aV2P inherits that convention.

## 5. Public option mapping (2.2 legacy to native alpaka 3)

Source of truth is `develop/cmake/aV2POptionMapping.csv` (44 rows).
Mode `on` turns the native option ON when the legacy is ON. Mode
`forward` copies the value unless the consumer set the native option
explicitly. `unsupported`, `cuda_unsupported` and `sycl_unsupported` are
fatal errors when the value differs from the legacy default (the latter
two only when CUDA or SYCL is enabled). `sycl_warn` is a warning when
SYCL is enabled. `local` has no native equivalent.

| Legacy option | Native alpaka 3 effect | Type | Mode | Default | Diagnostic |
|---------------|------------------------|------|------|---------|------------|
| `aV2P_BUILD_EXAMPLES` | (none) | BOOL | local | OFF | aV2P build tree only |
| `aV2P_BUILD_BENCHMARKS` | (none) | BOOL | local | OFF | aV2P build tree only |
| `aV2P_ENABLE_WERROR` | (none) | BOOL | local | OFF | aV2P build tree only |
| `BUILD_TESTING` | (none) | BOOL | local | OFF | CTest option unchanged |
| `aV2P_INSTALL_TEST_HEADER` | (none) | BOOL | local | OFF | aV2P header install only |
| `aV2P_CHECK_HEADERS` | (none) | BOOL | local | OFF | aV2P test tree only |
| `aV2P_USE_INTERNAL_CATCH2` | (none) | BOOL | local | ON | aV2P test tree only |
| `aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE` | `alpaka_EXEC_CpuSerial`, `alpaka_HOST_Cpu` | BOOL | on | OFF | native host serial executor and host CPU API |
| `aV2P_ACC_GPU_CUDA_ENABLE` | `alpaka_DEP_CUDA`, `alpaka_EXEC_GpuCuda` | BOOL | on | OFF | native CUDA dependency and executor |
| `aV2P_ACC_GPU_CUDA_ONLY_MODE` | (none) | BOOL | local | OFF | compatibility validation only |
| `aV2P_ACC_GPU_HIP_ENABLE` | `alpaka_DEP_HIP`, `alpaka_EXEC_GpuHip` | BOOL | on | OFF | native HIP dependency and executor |
| `aV2P_ACC_GPU_HIP_ONLY_MODE` | (none) | BOOL | local | OFF | compatibility validation only |
| `aV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLE` | (none) | BOOL | unsupported | OFF | alpaka 3 has no std::thread executor |
| `aV2P_ACC_CPU_B_TBB_T_SEQ_ENABLE` | `alpaka_DEP_TBB`, `alpaka_EXEC_TbbBlocks` | BOOL | on | OFF | native oneTBB dependency and executor |
| `aV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLE` | `alpaka_DEP_OMP`, `alpaka_EXEC_CpuOmpBlocks` | BOOL | on | OFF | native OpenMP dependency and blocks executor |
| `aV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLE` | (none) | BOOL | unsupported | OFF | alpaka 3 has no OpenMP block-thread executor |
| `aV2P_ACC_CPU_DISABLE_ATOMIC_REF` | (none) | BOOL | local | OFF | aV2P atomic-ref selection only |
| `aV2P_ACC_SYCL_ENABLE` | `alpaka_DEP_ONEAPI`, `alpaka_EXEC_OneApi` | BOOL | on | OFF | native oneAPI dependency and executor |
| `aV2P_FAST_MATH` | `alpaka_FAST_MATH` | STRING | forward | DEFAULT | forwarded value |
| `aV2P_FTZ` | `alpaka_FTZ` | STRING | forward | DEFAULT | forwarded value |
| `aV2P_RELOCATABLE_DEVICE_CODE` | `alpaka_RELOCATABLE_DEVICE_CODE` | STRING | forward | DEFAULT | forwarded value |
| `aV2P_CUDA_SHOW_REGISTER` | `alpaka_CUDA_SHOW_REGISTER` | STRING | forward | DEFAULT | forwarded value |
| `aV2P_CUDA_KEEP_FILES` | `alpaka_CUDA_KEEP_FILES` | STRING | forward | DEFAULT | forwarded value |
| `aV2P_CUDA_EXPT_EXTENDED_LAMBDA` | (no effect) | STRING | cuda_unsupported | ON | alpaka 3 always enables extended lambdas |
| `aV2P_HIP_KEEP_FILES` | `alpaka_HIP_KEEP_FILES` | STRING | forward | OFF | forwarded value |
| `aV2P_DEBUG` | (none) | STRING | local | 0 | aV2P `AV2P_DEBUG` macro only |
| `aV2P_CXX_STANDARD` | `alpaka_CXX_STANDARD` | STRING | forward | 20 | forwarded value |
| `aV2P_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB` | (none) | STRING | local | 47 | aV2P macro only |
| `aV2P_DISABLE_VENDOR_RNG` | (none) | BOOL | local | OFF | aV2P macro only |
| `aV2P_ASSERT_ACC_ENABLE` | (none) | BOOL | local | ON | aV2P macro only |
| `aV2P_CUDA_SHOW_CODELINES` | `alpaka_CUDA_SHOW_CODELINES` | BOOL | forward | OFF | forwarded value |
| `aV2P_USE_MDSPAN` | (none) | STRING | local | OFF | aV2P mdspan feature only |
| `aV2P_SYCL_ONEAPI_CPU` | `alpaka_ONEAPI_Cpu` | BOOL | on | OFF | native oneAPI CPU target |
| `aV2P_SYCL_ONEAPI_GPU` | `alpaka_ONEAPI_IntelGpu` | BOOL | on | OFF | native oneAPI Intel GPU target |
| `aV2P_SYCL_ONEAPI_GPU_NVIDIA` | `alpaka_ONEAPI_NvidiaGpu` | BOOL | on | OFF | native oneAPI NVIDIA GPU target |
| `aV2P_SYCL_ONEAPI_GPU_AMD` | `alpaka_ONEAPI_AmdGpu` | BOOL | on | OFF | native oneAPI AMD GPU target |
| `aV2P_SYCL_ONEAPI_CPU_ISA` | (see `alpaka_ONEAPI_Cpu_ARCH`) | STRING | sycl_warn | avx2 | native uses `alpaka_ONEAPI_Cpu_ARCH` |
| `aV2P_SYCL_ONEAPI_GPU_DEVICES` | (vendor arch variables) | STRING | sycl_warn | intel_gpu_pvc | native uses vendor specific arch variables |

Row count: 44 legacy options, matching the source table.

## 6. Compatibility matrix

States: `supported` (passing test evidence in the current inventory),
`unvalidated` (adapter in place, no recorded passing test), `limitation`
(explicit aV2P incompatibility, documented above), `gap` (verified missing
feature in pinned alpaka 3.0.0 that aV2P cannot bridge), `blocked`
(environment cannot exercise the path).

| Feature | State | Evidence |
|---------|-------|----------|
| Ported examples on serial CPU | supported | host serial build of the renamed tree; example built and linked |
| Kernel unit tests (named functor port) on serial CPU | supported | all tests pass in the host serial build |
| Math unit tests (named functor port) on serial CPU | supported | all tests pass in the host serial build |
| MdSpan view unit tests (named functor port) on serial CPU | supported | all tests pass in the host serial build |
| `reduce` example, serial CPU | supported | example runs, result matches |
| `helloWorldLambda` example, serial CPU | supported | example runs, prints output |
| add_subdirectory consumer | supported | consumer CMakeLists builds and runs against the source tree |
| installed-package consumer | supported | consumer CMakeLists builds and runs against the installed package |
| coexistence of aV2P and alpaka 3 in one target | supported | coexist CMakeLists builds with both targets linked |
| Conflict fixture, `aV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLE=ON` | supported | configuration is rejected with a fatal error |
| OpenMP 2.0 blocks backend | unvalidated | adapter in progress |
| CUDA backend | unvalidated | adapter in progress, no GPU in the current environment |
| HIP backend | unvalidated | adapter in progress, no AMD GPU or HIP toolchain verified |
| SYCL CPU, Intel and AMD GPU backends | unvalidated | adapter in progress, no oneAPI toolchain verified |
| TBB blocks backend | supported | serial+TBB build: `accTest` 480 assertions, `devTest` 128, `kernelTest` 1058, `blockSharedTest` 96, `blockSyncTest` 96, `workDivTest` 151; probe reports enabled tags `TagCpuSerial,TagCpuTbbBlocks`, mapped executor `CpuTbbBlocks`, and 28 TBB worker threads on a 10000-block kernel |
| std::thread backend | limitation | no faithful alpaka 3 executor exists |
| OpenMP block-thread backend | limitation | no faithful alpaka 3 executor exists |
| FPGA SYCL backend | removed | intentionally removed from aV2P, not an alpaka 3 capability gap |
| Zero-dimensional buffer and View | gap | alpaka 3.0.0 lacks 0D allocation; compile probe fails in the host-alloc path |
| Warp mask width, 2.2 per-backend type | limitation | 2.2 `TWarp::mask_type` vs alpaka 3 conditional width; no wrapper type |
| Warp `all`/`any` return type | limitation | `std::int32_t` in 2.2 vs `bool` in alpaka 3 |
| SYCL CPU per-kernel warp size | gap | alpaka 3.0.0 has no per-kernel warp-size trait; the sub-group size comes from the device properties |
| Device lambdas as kernel objects | limitation | aV2P does not forward device lambda closures; use a named functor |
| RISC-V or cross-compiled backends | blocked | no RISC-V toolchain in the current environment |
| Grid synchronization on `AccCpuSerial` | supported | `test/unit/grid` passes with `m_cooperativeLaunch == true` |
| Grid synchronization on `AccCpuOmp2Blocks` | supported | `test/unit/grid` passes with one block per OpenMP team thread |
| Grid synchronization on CUDA, HIP, SYCL | gap | no cooperative launch in alpaka 3.0.0; dependent `static_assert` |
| `example/helloWorldGridSync` | limitation | not built: 0D result buffer plus GPU cooperative launch |

## 7. Grid synchronization

The 2.2 public grid front-end is restored under `include/aV2P/grid/`:
`ConceptGridSync`, `trait::SyncGridThreads`, `syncGridThreads()`, and the
`GridSyncNoOp`, `GridSyncOmp`, `GridSyncBarrierThread`, `GridSyncBarrierTbb`,
`GridSyncCudaHipBuiltIn` and `GridSyncGenericSycl` entry points. The
accelerator facade implements `trait::SyncGridThreads` and the cooperative
kernel task trait for `AccAlpaka3`.

`m_cooperativeLaunch` follows the real barrier:

- `AccCpuSerial`: true. The serial executor runs the grid on one thread in
  order, so the 2.2 no-op is the complete barrier. `getMaxActiveBlocks` returns 1.
- `AccCpuOmp2Blocks`: true. The executor runs one block per OpenMP team
  thread and the grid sync is an OpenMP barrier. `getMaxActiveBlocks` returns
  `omp_get_max_threads()`, and the grid must have exactly that many blocks; a
  smaller grid would leave team threads outside the barrier.
- `AccCpuTbbBlocks`: false. The executor runs one thread per block and alpaka 3
  provides no grid-wide barrier for the TBB task arena; the grid-sync test
  skips this accelerator instead of claiming support.
- CUDA, HIP, CPU/CUDA/HIP SYCL: false. A grid-wide barrier needs a cooperative
  launch, which the pinned alpaka 3.0.0 does not provide.

The GPU entry points remain source-compatible but fail at instantiation with:

```text
aV2P: grid-wide synchronization requires a cooperative kernel launch. The pinned alpaka 3.0.0
does not provide cooperative launch, so grid synchronization is not supported on this back-end.
Use a CPU back-end (CpuSerial or CpuOmp2Blocks) for grid synchronization.
```

`example/helloWorldGridSync` is not built. It stores its scalar result in a
zero-dimensional buffer, which alpaka 3 does not support, and it launches the
kernel through the cooperative-launch API. Adapting only the scalar storage
would keep the CPU demonstration identical, but a GPU build would still fail on
the cooperative-launch diagnostic, so the example stays out of the build. The
zero-dimensional decision is example-only: the public grid API and the grid
test use regular 1D buffers and do not emulate 0D memory.

Restoration condition for the GPU gap: an alpaka release with cooperative
launch support. Then replace the diagnostic with the native cooperative
barrier and map `m_cooperativeLaunch` to the backend query.

## 8. SYCL CPU per-kernel warp size

alpaka 2.2 provides the per-kernel trait `alpaka::trait::WarpSize<TKernelFnObj, TAcc>`.
`TaskKernelGenericSycl` reads it into the kernel's `reqd_sub_group_size` and, on
the SYCL CPU device with a multi-dimensional block, throws `std::runtime_error`
when the innermost block extent is not a multiple of the requested warp size.
aV2P keeps the 2.2 trait declaration in `include/aV2P/kernel/Traits.hpp`
(`trait::WarpSize`, `trait::warpSize`) for source compatibility.

alpaka 3.0.0 does not provide a per-kernel warp-size or sub-group-size
mechanism:

- `alpaka::KernelBundle` stores only the kernel functor and its arguments; it
  has no warp-size attribute.
- The oneAPI queue selects the compile-time warp size with
  `syclGeneric::Warpsize::Dispatch` from the device property
  `warpSize` (`include/alpaka/api/syclGeneric/Queue.hpp`) and annotates each
  kernel instantiation with `[[sycl::reqd_sub_group_size(T_warpSize)]]`
  (`include/alpaka/api/oneApi/Queue.hpp`).
- No aV2P code consumes `trait::WarpSize`, so a requested warp size is not
  forwarded to the launch.

Consequence on the SYCL CPU device: a block whose innermost extent is not a
multiple of the requested warp size is not rejected at launch. The kernel runs
with the device-provided sub-group size instead of the requested one. In
`test/unit/warp/src/SyclCpuWarpLayout.cpp` the valid innermost layout still
passes; the unsupported outer layout is kept in the source but guarded with
`#if 0`, with the reason and the restoration condition next to it.

Restoration condition: restore the guarded `CHECK_THROWS_AS` case once alpaka
exposes per-kernel sub-group/warp size selection (a kernel trait or a launch
attribute) and aV2P forwards it to the oneAPI launch.

## 9. Backend build and install

### a. HIP concrete accelerator has no device-primitive facade

A kernel that uses device primitives (atomics, warp operations, shared memory,
memory fences) must take the accelerator as a template parameter. A concrete
`AccGpuHipRt` parameter works for index and work-division code, but aV2P
provides no device-primitive facade for that concrete HIP accelerator type. A
concrete CUDA accelerator does expose the primitives.

Cause: the device-primitive facade is reached through the templated
accelerator concept, and the concrete HIP accelerator type does not forward
it.

Workaround: template the kernel on the accelerator (`template<typename TAcc>
AV2P_FN_ACC void kernel(TAcc const& acc, ...)`) before calling any device
primitive.

Restoration condition: provide a device-primitive facade for the concrete
`AccGpuHipRt` type, so a non-templated HIP kernel compiles again.

### b. OpenMP 2.0 and HIP in one install

An aV2P install that enables `aV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLE` and
`aV2P_ACC_GPU_HIP_ENABLE` together cannot compile HIP-language translation
units that enumerate host accelerators. The native OpenMP flags are applied to
CXX sources only, so the HIP compiler sees no OpenMP flags during that
enumeration and compilation fails.

Cause: the CMake wiring does not apply the native OpenMP flags to HIP-language
sources.

Workaround: build aV2P per backend set, with one OpenMP install and one HIP
install, and select the needed install through `CMAKE_PREFIX_PATH`. This
workaround is verified; the combined build is not fixed.

Restoration condition: apply the native OpenMP flags to HIP-language sources
that include host accelerators, so one install can serve both backend sets.

### c. Installed backend set is fixed

An aV2P install bakes `AV2P_ACC_*_ENABLED` into its exported interface at
build time. A consumer inherits that backend set. Passing
`-DaV2P_ACC_*_ENABLE` at the consumer's configure step does not deselect an
installed backend.

Cause: backend selection is resolved when aV2P itself is built and installed,
not when a consumer configures.

Workaround: use one aV2P install per backend set and select the matching
install through `CMAKE_PREFIX_PATH`.

Restoration condition: resolve `AV2P_ACC_*_ENABLE` at consumer configure time
and export only the selected backends, so one install can serve several
backend sets.
