#!/usr/bin/env bash
#
# SPDX-License-Identifier: MPL-2.0
#
# Local and remote packaging smoke test for aV2P.
#
# It exercises:
#   1. add_subdirectory consumption of the aV2P source tree
#   2. installation of aV2P and installed find_package(aV2P CONFIG) consumption
#   3. a consumer that uses aV2P and native alpaka 3 in one translation unit
#   4. a negative conflict test: aV2P_ACC_GPU_CUDA_ENABLE=ON with alpaka_DEP_CUDA=OFF
#
# No GPU is required. The native alpaka dependency is reused from either
# --alpaka3-prefix (installed package) or --alpaka3-source (source tree, which is
# also installed into the work directory for the installed-consumer tests).
#
# Examples:
#   script/av2p-package-test.sh --alpaka3-source /path/to/alpaka-3.0.0
#   script/av2p-package-test.sh --alpaka3-prefix /opt/alpaka-3.0.0
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_SOURCE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

SOURCE_DIR="${DEFAULT_SOURCE_DIR}"
ALPAKA3_SOURCE=""
ALPAKA3_PREFIX=""
WORK_DIR=""
JOBS="${JOBS:-}"

usage()
{
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die()
{
    echo "error: $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --source-dir) SOURCE_DIR="$2"; shift 2 ;;
        --alpaka3-source) ALPAKA3_SOURCE="$2"; shift 2 ;;
        --alpaka3-prefix) ALPAKA3_PREFIX="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument '$1'" ;;
    esac
done

[ -f "${SOURCE_DIR}/CMakeLists.txt" ] || die "no aV2P source tree at ${SOURCE_DIR}"
[ -n "${ALPAKA3_SOURCE}${ALPAKA3_PREFIX}" ] || die "pass --alpaka3-source or --alpaka3-prefix"
[ -z "${ALPAKA3_SOURCE}" ] || [ -d "${ALPAKA3_SOURCE}" ] || die "alpaka3 source not found: ${ALPAKA3_SOURCE}"

if [ -z "${WORK_DIR}" ]; then
    WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/av2p-package-test.XXXXXX")
fi
mkdir -p "${WORK_DIR}"
WORK_DIR=$(cd "${WORK_DIR}" && pwd)

if [ -z "${JOBS}" ]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS=$(nproc)
    else
        JOBS=4
    fi
fi

run()
{
    echo "+ $*"
    "$@"
}

#-------------------------------------------------------------------------------
# Resolve the native alpaka package directory and prepare a prefix for consumers.

ALPAKA3_CONFIG=""
if [ -n "${ALPAKA3_SOURCE}" ]; then
    ALPAKA3_PREFIX="${WORK_DIR}/alpaka3-prefix"
    echo "== installing native alpaka 3 from ${ALPAKA3_SOURCE} into ${ALPAKA3_PREFIX}"
    run cmake -S "${ALPAKA3_SOURCE}" -B "${WORK_DIR}/alpaka3-build" \
        -DCMAKE_INSTALL_PREFIX="${ALPAKA3_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release
    run cmake --install "${WORK_DIR}/alpaka3-build"
fi

if [ -f "${ALPAKA3_PREFIX}/alpakaConfig.cmake" ]; then
    ALPAKA3_CONFIG="${ALPAKA3_PREFIX}"
else
    ALPAKA3_CONFIG=$(find "${ALPAKA3_PREFIX}" -path "*/cmake/alpaka/alpakaConfig.cmake" -print -quit 2>/dev/null || true)
    if [ -z "${ALPAKA3_CONFIG}" ]; then
        die "no alpakaConfig.cmake found below ${ALPAKA3_PREFIX}"
    fi
    ALPAKA3_CONFIG=$(dirname "${ALPAKA3_CONFIG}")
fi

if [ -n "${ALPAKA3_SOURCE}" ]; then
    ALPAKA3_BUILD_ARG="-DaV2P_ALPAKA3_SOURCE_DIR=${ALPAKA3_SOURCE}"
else
    ALPAKA3_BUILD_ARG="-DaV2P_ALPAKA3_DIR=${ALPAKA3_CONFIG}"
fi

#-------------------------------------------------------------------------------
echo "== add_subdirectory consumer"
run cmake -S "${SOURCE_DIR}/test/package/add_subdirectory" -B "${WORK_DIR}/add-subdirectory-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAV2P_SOURCE_DIR="${SOURCE_DIR}" \
    "${ALPAKA3_BUILD_ARG}"
run cmake --build "${WORK_DIR}/add-subdirectory-build" --parallel "${JOBS}"
run "${WORK_DIR}/add-subdirectory-build/aV2P-add-subdirectory"

#-------------------------------------------------------------------------------
echo "== install aV2P"
run cmake -S "${SOURCE_DIR}" -B "${WORK_DIR}/av2p-build" \
    -DCMAKE_INSTALL_PREFIX="${WORK_DIR}/av2p-prefix" \
    -DCMAKE_BUILD_TYPE=Release \
    -DaV2P_BUILD_EXAMPLES=OFF \
    -DaV2P_BUILD_BENCHMARKS=OFF \
    -DBUILD_TESTING=OFF \
    -DaV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
    "${ALPAKA3_BUILD_ARG}"
run cmake --build "${WORK_DIR}/av2p-build" --parallel "${JOBS}" --target install

#-------------------------------------------------------------------------------
echo "== installed find_package(aV2P) consumer"
run cmake -S "${SOURCE_DIR}/test/package/consumer" -B "${WORK_DIR}/consumer-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${WORK_DIR}/av2p-prefix;${ALPAKA3_PREFIX}" \
    -Dalpaka_DIR="${ALPAKA3_CONFIG}"
run cmake --build "${WORK_DIR}/consumer-build" --parallel "${JOBS}"
run "${WORK_DIR}/consumer-build/aV2P-consumer"

#-------------------------------------------------------------------------------
echo "== installed aV2P + native alpaka coexistence consumer"
run cmake -S "${SOURCE_DIR}/test/package/coexist" -B "${WORK_DIR}/coexist-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${WORK_DIR}/av2p-prefix;${ALPAKA3_PREFIX}" \
    -Dalpaka_DIR="${ALPAKA3_CONFIG}"
run cmake --build "${WORK_DIR}/coexist-build" --parallel "${JOBS}"
run "${WORK_DIR}/coexist-build/aV2P-coexist"

#-------------------------------------------------------------------------------
echo "== negative conflict test"
CONFLICT_LOG="${WORK_DIR}/conflict.log"
set +e
cmake -S "${SOURCE_DIR}/test/package/conflict" -B "${WORK_DIR}/conflict-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAV2P_SOURCE_DIR="${SOURCE_DIR}" \
    -DaV2P_ACC_GPU_CUDA_ENABLE=ON \
    -Dalpaka_DEP_CUDA=OFF \
    "${ALPAKA3_BUILD_ARG}" >"${CONFLICT_LOG}" 2>&1
CONFLICT_STATUS=$?
set -e
if [ "${CONFLICT_STATUS}" -eq 0 ]; then
    echo "error: conflict fixture configured successfully" >&2
    exit 1
fi
grep -q "Conflicting backend configuration" "${CONFLICT_LOG}" \
    || { cat "${CONFLICT_LOG}" >&2; die "conflict message missing"; }
grep -q "aV2P_ACC_GPU_CUDA_ENABLE" "${CONFLICT_LOG}" \
    || { cat "${CONFLICT_LOG}" >&2; die "legacy option missing from conflict message"; }
grep -q "alpaka_DEP_CUDA" "${CONFLICT_LOG}" \
    || { cat "${CONFLICT_LOG}" >&2; die "native option missing from conflict message"; }
echo "conflict test passed (see ${CONFLICT_LOG})"

#-------------------------------------------------------------------------------
echo "== reconfigure aV2P without re-downloading native alpaka"
run cmake -S "${SOURCE_DIR}" -B "${WORK_DIR}/av2p-build" \
    -DaV2P_ALPAKA3_OFFLINE=ON

echo "PASS: aV2P package tests successful in ${WORK_DIR}"
