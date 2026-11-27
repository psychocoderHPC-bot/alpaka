# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
# This recipe is maintained in the alpaka repository. The canonical copy lives
# in the Spack package repository. See spack/README.md for the synchronization
# procedure.

import glob
import os

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import (
    InstallError,
    any_combination_of,
    conflicts,
    depends_on,
    join_path,
    license,
    maintainers,
    requires,
    variant,
    version,
    when,
)


class Alpaka(CMakePackage, CudaPackage):
    """Abstraction Library for Parallel Kernel Acceleration."""

    homepage = "https://github.com/alpaka-group/alpaka"
    url = "https://github.com/alpaka-group/alpaka/archive/refs/tags/0.9.0.tar.gz"
    git = "https://github.com/alpaka-group/alpaka.git"

    maintainers("vvolkl")

    license("MPL-2.0-no-copyleft-exception")

    version("develop", branch="develop")
    version("dev_v3.x", branch="dev_v3.x")
    version("3.0.0", sha256="9afd1c14f445127aeeca9f29b5d17a33cdb0b9e974418a96d9f880b370b854f2")
    version("2.2.0", sha256="b38dc3f461da261a746956bd84ae85c62a9b3ef13c078c94bceb9783ae94d1f2")
    version("2.1.1", sha256="2d30a43594c55067297947b0ec83300e4f2899497464c5cc6f142c823f3ea1b2")
    # 2.1.0 is deprecated due to the buffer/view arrow operator bug
    # see https://github.com/alpaka-group/alpaka/pull/2600
    version(
        "2.1.0",
        sha256="e5de511561d7630e856e58b6e191e054f627938d4be70cfefdc47c388449d77f",
        deprecated=True,
    )
    version("1.3.0", sha256="8caec8de11a5537c721d2112c97252f06ffee709392ea02fcf62df4b50511714")
    version("2.0.0", sha256="ed313117aa922ef7260ec37bc5f79d750ae5547f0b9e0380a016590aa3a98e8b")
    version("1.2.0", sha256="069ea68ac950b17cffb3a3e790973aa5115f07ab23c0247a167e815b3c6e6fa2")
    version("1.1.0", sha256="95a8f1b706105d8a213116b6ba00e27bd904855c377f5a22a04aa0b86054dc35")
    version("1.0.0", sha256="38223dc1ca5bcf3916ff91f8825fb8caab7047430877222847e0ceb93bffecc9")
    version("0.9.0", sha256="3b2a5631366619fab5f3ceaf860219362f35db6c1148a601a3779a836cf29363")

    # Shared declarations
    variant("examples", default=False, description="Build alpaka examples")

    depends_on("cxx", type="build")  # generated
    depends_on("cmake@3.18:")
    depends_on("cmake@3.22:", when="@1:2")
    depends_on("cmake@3.25:", when="@2:2")
    depends_on("cmake@3.25:", when="@3:,dev_v3.x")

    variant(
        "boost_atomic_ref",
        default=False,
        when="@2:,dev_v3.x",
        description="Provide Boost as a fallback when C++20 std::atomic_ref is not available",
    )
    depends_on("boost@1.78:+atomic", when="@2:,dev_v3.x +boost_atomic_ref")

    # alpaka 2.x and older. The lower bound excludes the named development version:
    # Spack orders unknown version strings below numeric versions, therefore a plain
    # "@:2" would also match dev_v3.x.
    with when("@0.9:2"):
        variant(
            "backend",
            multi=True,
            values=(
                "serial",
                "threads",
                "tbb",
                "omp2_gridblock",
                "omp2_blockthread",
                "cuda",
                "cuda_only",
                "hip",
                "hip_only",
            ),
            description="Backends to enable",
            default="serial",
        )
        depends_on("boost@1.74:")

        # make sure no other backend is enabled if using cuda_only or hip_only
        for v in ("serial", "threads", "tbb", "omp2_gridblock", "omp2_blockthread", "cuda", "hip"):
            conflicts("backend=cuda_only,%s" % v)
            conflicts("backend=hip_only,%s" % v)
        conflicts("backend=cuda_only,hip_only")

        # todo: add conflict between cuda 11.3 and gcc 10.3.0
        # see https://github.com/alpaka-group/alpaka/issues/1297

    # alpaka 3.0.0 is a released, unmodified header-only package. It has no support
    # for persisting an API selection in the installed CMake package.
    with when("@3.0.0"):
        conflicts(
            "+cuda",
            msg="alpaka 3.0.0 cannot persist a CUDA configuration. Use alpaka@dev_v3.x +cuda.",
        )

    # alpaka 3.1 and the development version
    with when("@3.1:,dev_v3.x"):
        variant("hip", default=False, description="Enable the HIP API and install a HIP toolchain")
        variant(
            "oneapi",
            default=False,
            description="Enable the oneAPI SYCL API and install the oneAPI toolchain",
        )
        variant(
            "openmp",
            default=True,
            when="~hip",
            description="Enable the OpenMP host executor",
        )
        # alpaka's HIP implementation requires a Clang host compiler for OpenMP.
        # Disable OpenMP by default for HIP installations and reject an explicit
        # +hip+openmp request unless the compiler is Clang.
        variant(
            "openmp",
            default=False,
            when="+hip",
            description="Enable the OpenMP host executor",
        )
        variant(
            "hwloc",
            values=("auto", "on", "off"),
            default="auto",
            description="hwloc policy: auto searches at consumer configure time, on requires hwloc, off disables it",
        )
        variant(
            "cuda_compiler",
            values=("nvcc", "clang"),
            default="nvcc",
            when="+cuda",
            description="Compiler used for CUDA device code",
        )
        variant(
            "amdgpu_target",
            description="AMD GPU architecture used for HIP device code",
            values=any_combination_of(*ROCmPackage.amdgpu_targets),
            when="+hip",
        )

        conflicts("+cuda", when="+hip", msg="CUDA and HIP are mutually exclusive")
        conflicts("+cuda", when="+oneapi", msg="CUDA and oneAPI SYCL are mutually exclusive")
        conflicts("+hip", when="+oneapi", msg="HIP and oneAPI SYCL are mutually exclusive")

        requires(
            "%clang",
            when="+cuda cuda_compiler=clang",
            msg="CUDA compiled with Clang requires Clang as the C++ compiler. Use %clang.",
        )
        requires(
            "+openmp %clang",
            when="+hip +openmp",
            msg="HIP with OpenMP requires a Clang C++ compiler to use compatible OpenMP runtimes.",
        )

        depends_on("hip@6:", when="+hip")
        depends_on("llvm-amdgpu", when="+hip")
        depends_on("intel-oneapi-compilers", when="+oneapi")
        depends_on("hwloc@2:", when="hwloc=on")

    def cmake_args(self):
        if self.spec.satisfies("@0.9:2"):
            return self._legacy_cmake_args()
        if self.spec.satisfies("@3.0.0"):
            return self._v3_0_0_cmake_args()
        if self.spec.satisfies("@3.1:,dev_v3.x"):
            return self._v3_1_cmake_args()
        raise InstallError(f"Unsupported alpaka version {self.version}")

    def _legacy_cmake_args(self):
        spec = self.spec
        args = []
        if spec.satisfies("backend=serial"):
            args.append(self.define("ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE", True))
        if spec.satisfies("backend=threads"):
            args.append(self.define("ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE", True))
        if spec.satisfies("backend=tbb"):
            args.append(self.define("ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE", True))
        if spec.satisfies("backend=omp2_gridblock"):
            args.append(self.define("ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE", True))
        if spec.satisfies("backend=omp2_blockthread"):
            args.append(self.define("ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE", True))
        if spec.satisfies("backend=cuda"):
            args.append(self.define("ALPAKA_ACC_GPU_CUDA_ENABLE", True))
        if spec.satisfies("backend=cuda_only"):
            args.append(self.define("ALPAKA_ACC_GPU_CUDA_ENABLE", True))
            args.append(self.define("ALPAKA_ACC_GPU_CUDA_ONLY_MODE", True))
        if spec.satisfies("backend=hip"):
            args.append(self.define("ALPAKA_ACC_GPU_HIP_ENABLE", True))
        if spec.satisfies("backend=hip_only"):
            args.append(self.define("ALPAKA_ACC_GPU_HIP_ENABLE", True))
            args.append(self.define("ALPAKA_ACC_GPU_HIP_ONLY_MODE", True))

        args.append(self.define_from_variant("alpaka_BUILD_EXAMPLES", "examples"))
        # need to define, as it is explicitly declared as an option by alpaka:
        args.append(self.define("BUILD_TESTING", self.run_tests))
        return args

    def _v3_0_0_cmake_args(self):
        return [
            self.define("alpaka_TESTS", self.run_tests),
            self.define_from_variant("alpaka_EXAMPLES", "examples"),
            self.define("BUILD_TESTING", self.run_tests),
        ]

    def _v3_1_cmake_args(self):
        spec = self.spec
        hwloc = spec.variants["hwloc"].value
        args = [
            self.define("alpaka_INSTALL_PIN_CONFIG", True),
            self.define("alpaka_DEP_OMP", "+openmp" in spec),
            self.define("alpaka_DEP_HWLOC", hwloc.upper()),
            self.define("alpaka_TESTS", self.run_tests),
            self.define_from_variant("alpaka_EXAMPLES", "examples"),
            self.define("BUILD_TESTING", self.run_tests),
        ]

        if "+cuda" in spec:
            args.extend(self._v3_cuda_args())
        if "+hip" in spec:
            args.extend(self._v3_hip_args())
        if "+oneapi" in spec:
            args.extend(self._v3_oneapi_args())
        return args

    def _v3_cuda_args(self):
        spec = self.spec
        cuda_prefix = spec["cuda"].prefix
        mode = spec.variants["cuda_compiler"].value
        if mode == "nvcc":
            compiler = join_path(cuda_prefix.bin, "nvcc")
        else:
            compiler = self.compiler.cxx

        args = [
            self.define("alpaka_DEP_CUDA", True),
            self.define("CMAKE_CUDA_COMPILER", compiler),
            self.define("CUDAToolkit_ROOT", cuda_prefix),
        ]

        archs = spec.variants["cuda_arch"].value
        if archs and "none" not in archs:
            args.append(self.define("CMAKE_CUDA_ARCHITECTURES", archs))

        if mode == "clang":
            args.append(self.define("CMAKE_CUDA_FLAGS", f"--cuda-path={cuda_prefix}"))
        return args

    def _v3_hip_args(self):
        spec = self.spec
        llvm_prefix = spec["llvm-amdgpu"].prefix
        compiler = None
        for candidate in (
            join_path(llvm_prefix.lib.llvm.bin, "clang++"),
            join_path(llvm_prefix.bin, "clang++"),
            join_path(llvm_prefix.bin, "amdclang++"),
            join_path(llvm_prefix.lib.llvm.bin, "amdclang++"),
        ):
            if os.path.isfile(candidate):
                compiler = candidate
                break
        if compiler is None:
            raise InstallError(
                f"Could not find a HIP-capable Clang compiler in the llvm-amdgpu prefix '{llvm_prefix}'."
            )
        args = [
            self.define("alpaka_DEP_HIP", True),
            self.define("CMAKE_HIP_COMPILER", compiler),
        ]

        archs = spec.variants["amdgpu_target"].value
        if archs and "none" not in archs:
            args.append(self.define("CMAKE_HIP_ARCHITECTURES", archs))
        return args

    def _v3_oneapi_args(self):
        compiler, intelsycl_dir = self._v3_oneapi_compiler_and_sycl_dir()
        return [
            self.define("alpaka_DEP_ONEAPI", True),
            self.define("CMAKE_CXX_COMPILER", compiler),
            self.define("IntelSYCL_DIR", intelsycl_dir),
        ]

    def _v3_oneapi_compiler_and_sycl_dir(self):
        prefix = str(self.spec["intel-oneapi-compilers"].prefix)
        roots = [prefix]
        roots.extend(sorted(glob.glob(os.path.join(prefix, "compiler", "*"))))
        for root in reversed(roots):
            compiler = os.path.join(root, "bin", "icpx")
            intelsycl_dir = os.path.join(root, "lib", "cmake", "IntelSYCL")
            if os.path.isfile(compiler) and os.path.isfile(os.path.join(intelsycl_dir, "IntelSYCLConfig.cmake")):
                return compiler, intelsycl_dir
        raise InstallError(
            "Could not find icpx and IntelSYCLConfig.cmake in the intel-oneapi-compilers prefix "
            f"'{prefix}'. Set the prefix to an installation that contains the oneAPI compiler."
        )
