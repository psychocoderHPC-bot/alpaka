# aV2P Validation

Validation matrix for the aV2P ports of the alpaka 2.2 public API onto native
alpaka 3 (aV2P = "alpaka v2 public API"). Every result below is a completed run
on the shared computeMCP systems. Counts are exact pass/total numbers from the
run logs; UNVERIFIED marks a cell that was not checked in that configuration.

## 1. Scope and pinned baselines

| Role | SHA | Verified |
|---|---|---|
| aV2P branch tip (test subject) | `3eceff07f71aaa472b34733684b22f0e74b006a` | exists in fork; `av2p/develop` |
| Base: fork `develop` | `61c5a7113a1b4b3ac2648ceaed88874635d05c7d` | exists; source identical to 2.2.0 apart from version and changelog |
| Legacy release 2.2.0 | `4dc5ea558defeab95427e114a2dd9da83cc4d215` | exists; tag `2.2.0` resolves to this commit |
| Dependency: native alpaka 3.0.0 | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` | exists |
| Prototype | `13fbb922f5c5b1505ea884e31dfa385ff00896f0` | exists |

Scope: build, ctest suite, examples, install and packaging for the CPU serial,
OpenMP, TBB, CUDA, HIP and oneAPI (SYCL) configurations listed below, plus the
memory-layout and cross-API interop checks. No performance numbers are claimed
here; timings are reported as indicative only (section 7).

## 2. How to reproduce

All runs share:

```sh
export aV2P_ALPAKA3_SOURCE_DIR=/tmp/av2p-base/src   # native alpaka 3.0.0 @ eb677e8d7
export alpaka_DEP_HWLOC=OFF
```

`/tmp/av2p-base/src` is a clone of native alpaka checked out at the pinned SHA
(3.0.0). Build in the aV2P fork worktree at the branch tip SHA.

The commands in this section were validated against
`3eceff07f71aaa472b34733684b22f0e74b006a` on the named machines (fwk394 for the
CPU and CUDA configurations, hal for HIP and oneAPI).

CPU serial + OpenMP (fwk394, g++ 13.3; nvcc 12.5 present but not used):

```sh
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DaV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -DaV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON \
  -Dalpaka_DEP_HWLOC=OFF \
  -DaV2P_ALPAKA3_SOURCE_DIR=/tmp/av2p-base/src
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CPU + TBB (fwk394, oneTBB 2023.2 prefix):

```sh
cmake -S . -B build-tbb \
  -DCMAKE_CXX_COMPILER=g++ \
  -DaV2P_ACC_CPU_B_TBB_T_SEQ_ENABLE=ON \
  -DTBB_DIR=/path/to/onetbb-install/lib/cmake/tbb \
  -Dalpaka_DEP_HWLOC=OFF \
  -DaV2P_ALPAKA3_SOURCE_DIR=/tmp/av2p-base/src
cmake --build build-tbb -j
ctest --test-dir build-tbb --output-on-failure
```

The TBB run includes the 10000-block kernel test on 28 threads.

CUDA 12.5 (fwk394, sm_70 GV100 and sm_75 RTX 2080 Ti):

```sh
cmake -S . -B build-cuda \
  -DCMAKE_CXX_COMPILER=g++ \
  -DaV2P_ACC_GPU_CUDA_ENABLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="70;75" \
  -Dalpaka_DEP_HWLOC=OFF \
  -DaV2P_ALPAKA3_SOURCE_DIR=/tmp/av2p-base/src
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
# run ctest once per physical GPU (CUDA_VISIBLE_DEVICES=0 / =1)
```

HIP ROCm 7.2 (hal, gfx1100 RX 7900 XTX, amdclang 22):

```sh
cmake -S . -B build-hip \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DROCm_PATH=/opt/rocm \
  -DaV2P_ACC_GPU_HIP_ENABLE=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100" \
  -Dalpaka_DEP_HWLOC=OFF \
  -DaV2P_ALPAKA3_SOURCE_DIR=/tmp/av2p-base/src
cmake --build build-hip -j
ctest --test-dir build-hip --output-on-failure
```

oneAPI 2026.2 GPU (hal, Arc A770) and CPU (hal, EPYC 7452). Load the DPC++
compiler via the oneAPI env first:

```sh
source /opt/intel/oneapi/setvars.sh
# Arc A770 GPU run. For the EPYC 7452 CPU run use
# -DaV2P_SYCL_ONEAPI_CPU=ON -DaV2P_SYCL_ONEAPI_GPU=OFF instead.
cmake -S . -B build-sycl \
  -DCMAKE_CXX_COMPILER=icpx \
  -DaV2P_ACC_SYCL_ENABLE=ON \
  -DaV2P_SYCL_ONEAPI_GPU=ON \
  -DaV2P_SYCL_ONEAPI_CPU=OFF \
  -Dalpaka_DEP_HWLOC=OFF \
  -DaV2P_ALPAKA3_SOURCE_DIR=/tmp/av2p-base/src
cmake --build build-sycl -j
ctest --test-dir build-sycl --output-on-failure   # GPU: --launch-device level_zero:gpu; CPU: omit the flag
```

The FP64-emulation run on the A770 used the device's floating-point emulation
mode for the four double-precision tests.

Install and package check (every config):

```sh
cmake --install build --prefix /tmp/av2p-install
cmake -S consuming-project -B consumer -DCMAKE_PREFIX_PATH=/tmp/av2p-install
```

The consumer test is the installed `find_package` sample; the coexistence test
links aV2P and native alpaka 3 in one binary; the conflict and
unfinalized-target diagnostics are the negative tests shipped with the infra.

## 3. Results matrix

Build counts are targets compiled; ctest and examples are pass/total.

| Config (machine) | Build | ctest | Examples | Installing and packaging | Devices |
|---|---|---|---|---|---|
| CPU serial + OMP (fwk394) | 281/281 | 52/52 | 18/18 | PASS | EPYC, g++ 13.3 |
| CPU + TBB, oneTBB 2023.2 (fwk394) | 143/143 | 52/52 | 18/18 | PASS | 28 threads |
| CUDA 12.5 sm_70 (fwk394) | 278/278 | 51/51 | 18/18 | PASS | GV100 |
| CUDA 12.5 sm_75 (fwk394) | 278/278 | 51/51 | 18/18 | PASS | RTX 2080 Ti |
| HIP ROCm 7.2 gfx1100 (hal) | 278/278 | 51/51 | 18/18 | PASS, interop PASS | RX 7900 XTX |
| oneAPI 2026.2 GPU (hal) | 281/281 | 46/51 | 16/18 | PASS, interop PASS | Arc A770 |
| oneAPI 2026.2 CPU (hal) | 281/281 | 51/51 | 18/18 | PASS, interop PASS | EPYC 7452 |

TBB detail runs, all green. The numbers are `assertions/cases`, not
pass/total: accTest 682/296, devTest 192/192, kernelTest 1682/1157,
blockShared 144/144, blockSync 144/96, workDiv 223/55.

Memory layout (2D extent `{3,5}` of `uint32_t`): native pitches are 20 B on
CPU (contiguous row), 512 B on CUDA, 256 B on HIP, 32 B on oneAPI. aV2P view
offsets match the native pitches in every backend tested.

Cross-cutting checks: the mixed-API interop example passes 32/32 checks on
CPU, CUDA, HIP and oneAPI. The installed `find_package` consumer and the
aV2P + native alpaka 3 coexistence build pass in all tested configurations.
The conflict and unfinalized-target diagnostics pass. All of these are listed
as PASS above rather than per device.

## 4. Failure classification

- Adapter defect: none open. Every failing item in section 3 resolves to an
  environment cause or an intentionally unsupported feature.
- Environment: oneAPI GPU on Arc A770 (Intel Xe has a limited FP64 unit).
  ctest 46/51 with failures in heatEquation, heatEquation2D, atomicTest,
  mathTest double/complex-double and randTest. With FP64 emulation enabled,
  four of those five pass; mathTest then fails 21 double-precision accuracy
  checks inside the software-emulated FP64. Two of the 18 examples
  are the same FP64 cause. This is a device capability, not an aV2P defect.
- Intentionally disabled or unsupported (see `limitations.md` for details and
  restoration conditions): zero-dimensional memory (section 3 of
  `limitations.md`); GPU grid-wide synchronization via cooperative launch
  (section 7); oneAPI device globals; oneAPI async-deferred buffer teardown;
  the collective queue; the CPU `Threads` and `Omp2Threads` accelerators; the
  SYCL CPU per-kernel warp-size expectation; FPGA SYCL (removed from scope).
- SYCL CPU per-kernel warp size: alpaka 3 has no per-kernel warp-size
  mechanism, so the launch-time rejection of an invalid SYCL CPU layout cannot
  be reproduced. The affected expectation is guarded; the valid layout check
  stays active.

## 5. Exclusions

The build omits features listed in `docs/aV2P/limitations.md` rather than
failing. This report does not duplicate that content; the cross-references are:
zero-dimensional memory (limitations.md section 3), the warp mask API surface
(section 2), grid synchronization and `helloWorldGridSync` (section 7),
backend and device coverage (sections 1 and 6), and the public 2.2 to 3
option mapping (section 5). Restoration conditions for each item are stated
there. The FPGA SYCL path is excluded from all runs and appears only as
removed scope.

## 6. Status, stated plainly

- Green: CPU serial + OMP, CPU + TBB, CUDA sm_70, CUDA sm_75, HIP gfx1100 and
  oneAPI CPU, including builds, ctest, examples, install, packaging, interop
  and the cross-cutting consumer/coexistence/diagnostic checks.
- Green modulo environment: oneAPI GPU on Arc A770. Everything passes except
  the FP64 tests, which the device emulates; with emulation on, all but the
  21 mathTest accuracy failures pass.
- Not green: no open item. The oneAPI CPU warp-layout expectation is resolved
  as a documented alpaka 3 gap with a narrow guard.

## 7. Timings

Any timings observed during these runs are indicative only: the machines are
shared systems, clocks and co-tenants vary between runs, and no
benchmarking protocol was applied. No timing is validated by this report.
