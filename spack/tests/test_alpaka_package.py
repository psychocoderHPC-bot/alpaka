# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0

"""Integration tests for the in-tree alpaka Spack recipe.

The fast tests only concretize specs and exercise the installed CMake package through a
plain CMake install. The Spack installation test is enabled with
ALPAKA_SPACK_INSTALL_TESTS=1 and is the same test used by the CPU-only CI job.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACK_REPO = REPO_ROOT / "spack" / "spack_repo" / "alpaka"
CONSUMER_DIR = REPO_ROOT / "spack" / "tests" / "consumer"


def spack_binary():
    if os.environ.get("SPACK"):
        return os.environ["SPACK"]
    if os.environ.get("SPACK_ROOT"):
        return str(Path(os.environ["SPACK_ROOT"]) / "bin" / "spack")
    return shutil.which("spack")


class Spack:
    def __init__(self, binary, env):
        self.binary = binary
        self.env = env
        self._root_spec_cache = {}
        self._dependency_cache = {}

    def run(self, *args, check=True, cwd=None):
        command = [self.binary, *args]
        result = subprocess.run(
            command,
            cwd=cwd,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if check and result.returncode != 0:
            raise AssertionError(f"{' '.join(command)} failed:\n{result.stdout}")
        return result

    def root_spec(self, spec, *options):
        result = self.run("spec", *options, "--format", "{name}@{version}{variants}", spec)
        return result.stdout.strip().splitlines()[-1]

    def dependency_names(self, spec):
        result = self.run("spec", spec)
        return {match.group(1) for match in re.finditer(r"\^([A-Za-z0-9_-]+)@", result.stdout)}


@pytest.fixture(scope="session")
def spack(tmp_path_factory):
    binary = spack_binary()
    if not binary:
        pytest.skip("Spack is not available; set SPACK or SPACK_ROOT")

    root = tmp_path_factory.mktemp("alpaka-spack")
    env = os.environ.copy()
    env["SPACK_USER_CONFIG_PATH"] = str(root / "config")
    env["SPACK_USER_CACHE_PATH"] = str(root / "cache")
    (root / "config").mkdir(parents=True)

    # Keep concretization fast and independent of GPU toolkits on the test machine.
    # These are only used to resolve accelerator specs; the installation test is host-only.
    prefixes = {}
    for package, version in (
        ("cuda", "12.5"),
        ("hip", "6.2.4"),
        ("llvm-amdgpu", "6.2.4"),
        ("intel-oneapi-compilers", "2026.0.0"),
    ):
        prefix = root / "prefixes" / package
        prefix.mkdir(parents=True)
        prefixes[package] = (version, prefix)

    packages = ["packages:"]
    for package, (version, prefix) in prefixes.items():
        packages.extend(
            [
                f"  {package}:",
                "    externals:",
                f"    - spec: {package}@{version}",
                f"      prefix: {prefix}",
                "    buildable: false",
            ]
        )
    (root / "config" / "packages.yaml").write_text("\n".join(packages) + "\n")

    driver = Spack(binary, env)
    driver.run("repo", "add", "--scope", "user", str(SPACK_REPO))
    driver.run("compiler", "find")
    driver.run("external", "find", "cmake", check=False)
    return driver


def build_and_run_consumer(build_dir, install_prefix, device="host", extra_args=()):
    command = [
        "cmake",
        "-S",
        str(CONSUMER_DIR),
        "-B",
        str(build_dir),
        f"-DCMAKE_PREFIX_PATH={install_prefix}",
        f"-DALPAKA_TEST_DEVICE={device}",
        *extra_args,
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise AssertionError(f"consumer configure failed:\n{result.stdout}")

    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", "4"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AssertionError(f"consumer build failed:\n{result.stdout}")

    result = subprocess.run(
        [str(build_dir / "alpaka_spack_consumer")],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AssertionError(f"consumer run failed:\n{result.stdout}")
    assert "OK" in result.stdout
    return result.stdout


def test_recipe_loads(spack):
    info = spack.run("info", "alpaka").stdout
    assert "dev_v3.x" in info
    assert "3.0.0" in info
    assert "2.2.0" in info


def test_legacy_specs_are_preserved(spack):
    assert "backend:=serial" in spack.root_spec("alpaka@2.2.0")
    assert "backend:=threads" in spack.root_spec("alpaka@2.2.0 backend=threads")
    assert "backend:=cuda_only" in spack.root_spec("alpaka@2.2.0 backend=cuda_only")
    assert "boost" in spack.dependency_names("alpaka@2.2.0")
    assert "boost" in spack.dependency_names("alpaka@2.2.0 +boost_atomic_ref")
    assert spack.root_spec("alpaka@2.1.0", "--deprecated").startswith("alpaka@2.1.0")
    assert spack.root_spec("alpaka@1.3.0").startswith("alpaka@1.3.0")


def test_generic_development_install_is_lean(spack):
    root = spack.root_spec("alpaka@dev_v3.x")
    assert "+openmp" in root
    assert "~cuda" in root
    assert "~hip" in root
    assert "~oneapi" in root
    assert "hwloc=auto" in root
    assert "backend" not in root

    dependencies = spack.dependency_names("alpaka@dev_v3.x")
    assert "cuda" not in dependencies
    assert "hip" not in dependencies
    assert "intel-oneapi-compilers" not in dependencies
    assert "hwloc" not in dependencies
    assert "boost" not in dependencies

    assert "~openmp" in spack.root_spec("alpaka@dev_v3.x ~openmp")
    assert "hwloc=on" in spack.root_spec("alpaka@dev_v3.x hwloc=on")
    assert "hwloc" in spack.dependency_names("alpaka@dev_v3.x hwloc=on")
    assert "hwloc=off" in spack.root_spec("alpaka@dev_v3.x hwloc=off")
    assert "hwloc" not in spack.dependency_names("alpaka@dev_v3.x hwloc=off")
    assert "boost" in spack.dependency_names("alpaka@dev_v3.x +boost_atomic_ref")


def test_release_3_0_0_is_generic_only(spack):
    root = spack.root_spec("alpaka@3.0.0")
    assert "~cuda" in root
    assert "openmp" not in root

    result = spack.run("spec", "alpaka@3.0.0 +cuda", check=False)
    assert result.returncode != 0
    assert "cannot persist" in result.stdout

    result = spack.run("spec", "alpaka@3.0.0 +hip", check=False)
    assert result.returncode != 0


def test_development_version_dispatches_to_new_implementation(spack):
    assert "+openmp" in spack.root_spec("alpaka@dev_v3.x")
    assert "openmp" not in spack.root_spec("alpaka@3.0.0")
    assert "+cuda" in spack.root_spec("alpaka@dev_v3.x +cuda cuda_arch=80")


@pytest.mark.skipif(
    os.environ.get("ALPAKA_SPACK_GPU_SPEC_TESTS") != "1",
    reason="set ALPAKA_SPACK_GPU_SPEC_TESTS=1 to run accelerator concretization tests",
)
def test_cuda_specs(spack):
    root = spack.root_spec("alpaka@dev_v3.x +cuda cuda_arch=80")
    assert "+cuda" in root
    dependencies = spack.dependency_names("alpaka@dev_v3.x +cuda cuda_arch=80")
    assert "cuda" in dependencies
    assert "hip" not in dependencies
    assert "intel-oneapi-compilers" not in dependencies

    result = spack.run("spec", "alpaka@dev_v3.x +cuda cuda_compiler=clang %gcc", check=False)
    assert result.returncode != 0
    assert "requires Clang" in result.stdout


@pytest.mark.skipif(
    os.environ.get("ALPAKA_SPACK_GPU_SPEC_TESTS") != "1",
    reason="set ALPAKA_SPACK_GPU_SPEC_TESTS=1 to run accelerator concretization tests",
)
def test_hip_specs(spack):
    root = spack.root_spec("alpaka@dev_v3.x +hip amdgpu_target=gfx1100")
    assert "+hip" in root
    assert "~openmp" in root
    assert "hip" in spack.dependency_names("alpaka@dev_v3.x +hip amdgpu_target=gfx1100")

    result = spack.run("spec", "alpaka@dev_v3.x +hip +openmp %gcc", check=False)
    assert result.returncode != 0
    assert "Clang" in result.stdout


@pytest.mark.skipif(
    os.environ.get("ALPAKA_SPACK_GPU_SPEC_TESTS") != "1",
    reason="set ALPAKA_SPACK_GPU_SPEC_TESTS=1 to run accelerator concretization tests",
)
def test_oneapi_specs(spack):
    root = spack.root_spec("alpaka@dev_v3.x +oneapi")
    assert "+oneapi" in root
    assert "intel-oneapi-compilers" in spack.dependency_names("alpaka@dev_v3.x +oneapi")


@pytest.mark.skipif(
    os.environ.get("ALPAKA_SPACK_GPU_SPEC_TESTS") != "1",
    reason="set ALPAKA_SPACK_GPU_SPEC_TESTS=1 to run accelerator concretization tests",
)
def test_accelerator_conflicts(spack):
    for spec in (
        "alpaka@dev_v3.x +cuda +hip",
        "alpaka@dev_v3.x +cuda +oneapi",
        "alpaka@dev_v3.x +hip +oneapi",
    ):
        result = spack.run("spec", spec, check=False)
        assert result.returncode != 0
        assert "mutually exclusive" in result.stdout


def test_cmake_pinned_install_persists_and_rejects(tmp_path):
    install_prefix = tmp_path / "alpaka-install"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT),
            "-B",
            str(tmp_path / "alpaka-build"),
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            "-Dalpaka_INSTALL_PIN_CONFIG=ON",
            "-Dalpaka_DEP_OMP=ON",
            "-Dalpaka_DEP_HWLOC=OFF",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert configure.returncode == 0, configure.stdout
    subprocess.run(
        ["cmake", "--install", str(tmp_path / "alpaka-build")],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    pinned = install_prefix / "lib" / "cmake" / "alpaka" / "alpakaConfigPinned.cmake"
    assert pinned.exists()
    assert 'alpaka_DEP_OMP "ON"' in pinned.read_text()

    output = build_and_run_consumer(tmp_path / "consumer-ok", install_prefix)
    assert "Host Cpu OK" in output

    with pytest.raises(AssertionError) as error:
        build_and_run_consumer(
            tmp_path / "consumer-rejected",
            install_prefix,
            extra_args=["-Dalpaka_DEP_OMP=OFF"],
        )
    assert "alpaka_DEP_OMP" in str(error.value)
    assert "already installed" in str(error.value) or "Remove the conflicting" in str(error.value)


@pytest.mark.skipif(
    os.environ.get("ALPAKA_SPACK_INSTALL_TESTS") != "1",
    reason="set ALPAKA_SPACK_INSTALL_TESTS=1 to run the Spack installation test",
)
def test_spack_installed_development_package(spack, tmp_path):
    env_dir = tmp_path / "env"
    spack.run("env", "create", str(env_dir))
    spack.run("-e", str(env_dir), "add", "alpaka@dev_v3.x")
    spack.run(
        "-e",
        str(env_dir),
        "develop",
        "-p",
        str(REPO_ROOT),
        "--no-clone",
        "alpaka@dev_v3.x",
    )
    spack.run("-e", str(env_dir), "install", "-y")

    env_yaml = (env_dir / "spack.yaml").read_text()
    assert str(REPO_ROOT) in env_yaml

    view = env_dir / ".spack-env" / "view"
    pinned = view / "lib" / "cmake" / "alpaka" / "alpakaConfigPinned.cmake"
    assert pinned.exists()
    assert 'alpaka_DEP_OMP "ON"' in pinned.read_text()

    output = build_and_run_consumer(tmp_path / "consumer-installed", view)
    assert "Host Cpu OK" in output

    with pytest.raises(AssertionError):
        build_and_run_consumer(
            tmp_path / "consumer-installed-rejected",
            view,
            extra_args=["-Dalpaka_DEP_OMP=OFF"],
        )
