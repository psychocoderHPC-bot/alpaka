# Spack package for alpaka

This directory contains the authoritative Spack recipe for alpaka. The recipe is
maintained in the alpaka repository and is periodically copied to the Spack
package repository.

## Repository layout

```text
spack/
├── README.md
├── spack_repo/
│   └── alpaka/
│       ├── repo.yaml
│       └── packages/
│           └── alpaka/
│               └── package.py
└── tests/
    ├── test_alpaka_package.py
    └── consumer/
```

The layout is a Spack package repository for Spack 1.2 and newer.

## Register the repository

With Spack 1.2:

```bash
spack repo add --scope user "$PWD/spack/spack_repo/alpaka"
spack info alpaka
```

The recipe is also usable from a checkout of the repository containing this
directory. The rest of this document assumes that the repository is registered.

## Tested versions

The integration tests use Spack `v1.2.0`
(`63962ed90680d339004305075da272ae93c04a41`). The builtin package repository
used by that Spack release is `releases/v2026.06`
(`46ab5375a9930f5943bda3c64ca85497f43295f8` at the time of writing). Spack
fetches the package repository automatically; pin it explicitly when a fully
reproducible installation is required.

## Supported versions

The recipe keeps the complete set of versions from the original Spack recipe,
including 0.9.0, 1.x, 2.0.0, 2.1.x, 2.2.0, and the deprecated 2.1.0 release.
The new versions are:

| Version | Source | Notes |
| --- | --- | --- |
| 3.0.0 | released tag | Unmodified release. Generic header-only installation only. |
| dev_v3.x | `dev_v3.x` branch | Development implementation with installed-configuration pinning. |
| develop | `develop` branch | Upstream 2.x development branch, kept for compatibility. |

Spack orders the named `dev_v3.x` version below numeric versions. The recipe
therefore uses explicit version unions:

* `@0.9:2` for the legacy 0.9 through 2.x implementation.
* `@3:,dev_v3.x` for all 3.x releases and the development branch.
* `@3.1:,dev_v3.x` for versions that support installed-configuration pinning
  (the unmodified 3.0.0 release does not).

Do not replace these with `@:2` or `@3:`: the plain ranges do not select the
intended implementation for `dev_v3.x`.

## Installation

### Legacy alpaka

Legacy versions keep the original `backend` variant and its CMake mapping:

```bash
spack install alpaka@2.2.0
spack install alpaka@2.2.0 backend=serial,threads
spack install alpaka@2.2.0 backend=cuda
spack install alpaka@2.2.0 +boost_atomic_ref
```

### Generic alpaka 3.x

```bash
spack install alpaka@3.0.0
spack install alpaka@dev_v3.x
```

A generic installation does not depend on CUDA, HIP, or oneAPI. The host API is
always enabled. The installed CMake package does not pin an accelerator API, so
different downstream builds can select different APIs from the same
installation:

```bash
# Host-only consumer
cmake -S consumer -B build-host

# CUDA consumer using the same alpaka installation
cmake -S consumer -B build-cuda \
  -Dalpaka_DEP_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

The consumer must select the API it uses. A generic installation deliberately
does not pin `alpaka_DEP_CUDA`, `alpaka_DEP_HIP`, or `alpaka_DEP_ONEAPI`.

The host configuration selected by the Spack variants is persisted. The default
generic installation pins OpenMP on and hwloc off:

```bash
spack install alpaka@dev_v3.x ~openmp
spack install alpaka@dev_v3.x +hwloc
```

### Pinned accelerator installations

```bash
spack install alpaka@dev_v3.x +cuda cuda_arch=90
spack install alpaka@dev_v3.x +cuda cuda_compiler=clang cuda_arch=90 %clang
spack install alpaka@dev_v3.x +hip amdgpu_target=gfx1100
spack install alpaka@dev_v3.x +oneapi
```

A pinned installation records the selected APIs, compiler, toolkit, and
architecture in `alpakaConfigPinned.cmake`. A downstream consumer cannot enable
an API that was not selected, disable a selected API, or replace the compiler.
Unsupported attempts fail during `find_package(alpaka)` with an actionable
error.

## Variants

### alpaka 3.1 and `dev_v3.x`

| Variant | Default | Meaning |
| --- | --- | --- |
| `+cuda` | off | Enable the CUDA API. Adds a `cuda` dependency. |
| `cuda_arch` | none | Value passed to `CMAKE_CUDA_ARCHITECTURES`. Set it for reproducible binaries. |
| `cuda_compiler=nvcc` | nvcc | Use `nvcc` for CUDA device code. |
| `cuda_compiler=clang` | | Use Clang for CUDA device code. Requires `%clang` as the host compiler. |
| `+hip` | off | Enable the HIP API. Adds `hip` and `llvm-amdgpu` dependencies. |
| `amdgpu_target` | none | Value passed to `CMAKE_HIP_ARCHITECTURES`. Set it for deployment. |
| `+oneapi` | off | Enable the oneAPI SYCL API. Adds the Intel oneAPI compiler. |
| `+openmp` | on, except with `+hip` | Enable the OpenMP host executor. |
| `+hwloc` | off | Require hwloc 2.x and enable the NUMA host API. |
| `+boost_atomic_ref` | off | Provide Boost as a fallback for `std::atomic_ref`. |
| `+examples` | off | Build the alpaka examples. |

`+hip+openmp` requires `%clang` because the HIP and host OpenMP runtimes must be
compatible. With `+hip` alone, OpenMP defaults to off. An explicit
`+hip+openmp %gcc` request fails during concretization.

CUDA, HIP, and oneAPI are mutually exclusive. HIP requires ROCm 6.0 or newer
(the upstream CMake code supports 6.0 through 10.0). oneAPI requires the Intel
`icpx` compiler.

### alpaka 3.0.0

The 3.0.0 release is installed unchanged. It supports a generic header-only
installation. The 3.1-only variants (`+hip`, `+oneapi`, `+openmp`, `+hwloc`,
`cuda_compiler`, `amdgpu_target`) are not available for 3.0.0. `+cuda` is
rejected because the release cannot persist the selection in the installed
CMake package. A consumer of a 3.0.0 installation can still select supported
APIs with `-Dalpaka_DEP_CUDA=ON` and similar options, provided the consumer
environment supplies the toolkits.

| Capability | 3.0.0 | `dev_v3.x` / 3.1 |
| --- | --- | --- |
| Generic header-only install | yes | yes |
| Legacy `backend` variants | no | no |
| Persist OpenMP/hwloc | no | yes |
| Persist accelerator API | no | yes |
| Persist CUDA/HIP compiler | no | yes |
| Persist architecture | no | yes |
| Reject conflicting consumer overrides | no | yes |
| `+hip`, `+oneapi` | no | yes |
| `cuda_compiler=clang` | no | yes |

Use `dev_v3.x` when pinning is required.

### alpaka 2.x and older

The original `backend` variant, its conflicts, the `examples` variant, and the
`boost_atomic_ref` variant are preserved. The `backend` values are `serial`,
`threads`, `tbb`, `omp2_gridblock`, `omp2_blockthread`, `cuda`, `cuda_only`,
`hip`, and `hip_only`.

## CUDA compiler selection

Both modes use the same `+cuda` variant. The compiler is selected with
`cuda_compiler`:

```bash
# nvcc
spack install alpaka@dev_v3.x +cuda cuda_arch=90
spack install alpaka@dev_v3.x +cuda cuda_arch=90 cuda_compiler=nvcc

# Clang CUDA, with Clang as the host compiler as well
spack install alpaka@dev_v3.x +cuda cuda_arch=90 cuda_compiler=clang %clang
```

For nvcc, the recipe passes the CUDA toolkit's `nvcc` as `CMAKE_CUDA_COMPILER`.
For Clang, it uses the concrete Clang compiler and passes the CUDA toolkit root
and `--cuda-path`. The installed configuration pins the compiler path and
toolkit root. Verify the result in a consumer build with:

```bash
grep CMAKE_CUDA_COMPILER_ID build/CMakeCache.txt
```

The pinned configuration rejects a consumer that configures with a different
CUDA compiler. For Clang CUDA, it also rejects a non-Clang `CMAKE_CXX_COMPILER`
after the C++ language has been enabled.

### Architecture

`cuda_arch` is mapped to `CMAKE_CUDA_ARCHITECTURES`. `amdgpu_target` is mapped
to `CMAKE_HIP_ARCHITECTURES`. When the variant is left at `none`, the recipe
does not set the CMake variable and the toolchain default is used. Set an
explicit value for installations that must not depend on the build machine.

## oneAPI notes

`+oneapi` installs the Intel oneAPI compiler package and configures
`CMAKE_CXX_COMPILER` to `icpx`. The installed CMake package validates that a
consumer uses the IntelLLVM compiler. Configure the consumer before `project()`
enables C++, for example:

```bash
cmake -S consumer -B build-oneapi \
  -DCMAKE_CXX_COMPILER=/path/to/oneapi/compiler/latest/bin/icpx \
  -DCMAKE_PREFIX_PATH=/path/to/alpaka
```

The oneAPI runtime libraries are provided by the compiler package. Source the
oneAPI environment (or use the Spack build environment) when running the
resulting executable.

## hwloc

`+hwloc` requires hwloc 2.x and pins `alpaka_DEP_HWLOC=ON`. The installed
configuration prepends the hwloc prefix and its `pkg-config` directory before
alpaka is configured. `~hwloc` pins `alpaka_DEP_HWLOC=OFF` and adds
`ALPAKA_DISABLE_HWLOC` to the host target. The generic default is `~hwloc` so a
plain alpaka installation does not acquire an ambient hwloc dependency.

## Downstream consumer

A minimal consumer needs `find_package(alpaka CONFIG REQUIRED)`, links
`alpaka::alpaka`, and calls `alpaka_finalize`:

```cmake
cmake_minimum_required(VERSION 3.25)
project(consumer LANGUAGES CXX)

find_package(alpaka CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE alpaka::alpaka)
alpaka_finalize(consumer)
```

For CUDA, enable the language after `find_package(alpaka)` and mark the source
as CUDA. For HIP, enable the HIP language and mark the source as HIP. For
oneAPI, configure with `icpx` as described above.

The installed package is header-only. `alpakaCommon.cmake` creates the targets
in the consumer build, so the pinned configuration is evaluated before language
enablement and before targets are created. Temporary build-tree paths are not
installed.

## Validation and tests

The tests use the Spack executable named by `SPACK` or `SPACK_ROOT`:

```bash
export SPACK=/path/to/spack/bin/spack

# Recipe loading, legacy concretization, generic 3.x, 3.0.0 dispatch,
# installed-configuration persistence, and consumer rejection tests.
pytest -q spack/tests/test_alpaka_package.py

# Generic development installation through Spack and a real consumer.
ALPAKA_SPACK_INSTALL_TESTS=1 pytest -q spack/tests/test_alpaka_package.py

# Accelerator concretization tests (no GPU required).
ALPAKA_SPACK_GPU_SPEC_TESTS=1 pytest -q spack/tests/test_alpaka_package.py
```

The tests create isolated Spack user configuration and cache directories and
use the repository checkout as the `spack develop` source. They do not test a
downloaded copy of an unrelated upstream branch.

GPU execution tests require the corresponding hardware and are not part of the
default CI job. The CPU-only subset is in `.github/workflows/spack.yml`.

## Maintenance and synchronization to Spack

There is exactly one recipe in the alpaka repository:
`spack/spack_repo/alpaka/packages/alpaka/package.py`.

To synchronize a change to the canonical Spack package repository:

1. Copy the recipe to `repos/spack_repo/builtin/packages/alpaka/package.py` in
   the Spack package repository.
2. Keep the Spack copyright and license header; the recipe is distributed under
   the same Apache-2.0/MIT terms as the Spack package repository.
3. Run `spack style` and `spack audit packages alpaka` in the Spack checkout.
4. Update checksums with `spack checksum alpaka` for new releases.
5. Remove any local-only changes from the copy, and do not maintain a second
   diverging recipe in this repository.

The alpaka maintainers should update the in-tree recipe first, run the tests,
and then send the synchronized copy to Spack.

## License

The recipe retains the Spack Project Developers copyright and the
`Apache-2.0 OR MIT` license. The rest of the alpaka repository is
MPL-2.0.
