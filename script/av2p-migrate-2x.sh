#!/usr/bin/env bash
#
# SPDX-License-Identifier: MPL-2.0
#
# Generic, idempotent alpaka 2.x -> aV2P migrator.
#
# The script applies only the mechanical renames documented in
# docs/aV2P/migration.md:
#   - C/C++ include paths   alpaka/ -> aV2P/
#   - C/C++ namespace       alpaka:: -> aV2P::
#   - C/C++ macro           ALPAKA_ -> AV2P_
#   - C/C++ macro           alpaka_ -> aV2P_
#   - CMake target          alpaka::alpaka -> aV2P::aV2P, alpaka:: -> aV2P::
#   - CMake package         find_package(alpaka ...) -> find_package(aV2P ...)
#   - CMake macro           ALPAKA_ -> aV2P_, alpaka_ -> aV2P_
#
# Device lambdas cannot be converted automatically. They are reported as
# manual work and never rewritten.
#
# Running the script twice is a no-op on the second run.
#
# Examples:
#   script/av2p-migrate-2x.sh --dry-run .
#   script/av2p-migrate-2x.sh --exclude 'external/*' /path/to/project
#   script/av2p-migrate-2x.sh --self-test
#
set -euo pipefail

SELF="${BASH_SOURCE[0]}"

DRY_RUN=0
QUIET=0
SELF_TEST=0
EXCLUDES=()
ROOTS=()

FILES_CHANGED=0
TOTAL_SUBS=0

CMAKE_SENTINEL="__AV2P_MIGRATE_CMAKE_TARGET__"

usage()
{
    cat <<'EOF'
Usage: av2p-migrate-2x.sh [options] [ROOT ...]

Rewrite the mechanical alpaka 2.x -> aV2P renames in ROOT (default: ".").
Only C/C++ source files and CMake files are touched. Device lambdas are
reported but never rewritten.

Options:
  --dry-run        Print a unified diff and write nothing. Exits 2 when the
                   migration would change at least one file, 0 otherwise.
  --exclude GLOB   Skip files whose path relative to ROOT matches GLOB.
                   Repeatable. Quote the glob to protect it from the shell.
  --no-backup      Accepted for compatibility. No backups are created.
  --quiet, -q      Suppress progress and the final summary.
                   Manual warnings are still printed.
  --self-test      Run an internal fixture test and exit.
  -h, --help       Show this help and exit.

Exit status:
  0  success, or a dry run without pending changes
  1  usage or I/O error
  2  dry run that would change files

Report lines and manual warnings are written to stdout; errors go to stderr.
EOF
}

die()
{
    echo "error: $*" >&2
    exit 1
}

warn_manual()
{
    printf 'manual   %s  %s\n' "$1" "$2"
}

count_matches()
{
    local pattern="$1"
    local file="$2"
    local result
    result=$(grep -E -o -- "$pattern" "$file" 2>/dev/null | wc -l) || true
    result="${result//[[:space:]]/}"
    printf '%s' "${result:-0}"
}

count_include_matches()
{
    local file="$1"
    local result
    result=$(grep -E '^[[:space:]]*#[[:space:]]*include' "$file" 2>/dev/null \
        | grep -E -o '[<"]alpaka/' | wc -l) || true
    result="${result//[[:space:]]/}"
    printf '%s' "${result:-0}"
}

report_device_lambdas()
{
    local file="$1"
    local display="$2"
    local ln
    while IFS= read -r ln; do
        [ -n "$ln" ] || continue
        warn_manual "${display}:${ln}" "device lambda requires conversion to a named functor"
    done < <(awk '
        { line[NR] = $0 }
        END {
            for (i = 1; i <= NR; ++i) {
                if (line[i] ~ /__global__|AV2P_FN_ACC/) {
                    if (line[i] ~ /\[[^]]*\]\(/ || line[i + 1] ~ /\[[^]]*\]\(/) {
                        print i
                    }
                }
            }
        }
    ' "$file")
}

finish_file()
{
    local path="$1"
    local display="$2"
    local tmp="$3"
    local subs="$4"

    if cmp -s -- "$path" "$tmp"; then
        rm -f -- "$tmp"
        return 0
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        diff -u --label "${display}" --label "${display}" "$path" "$tmp" || true
    else
        cp -- "$tmp" "$path"
    fi
    rm -f -- "$tmp"

    if [ "${QUIET}" -eq 0 ]; then
        printf 'changed  %s  (%s substitutions)\n' "${display}" "${subs}"
    fi

    FILES_CHANGED=$((FILES_CHANGED + 1))
    TOTAL_SUBS=$((TOTAL_SUBS + subs))
}

process_cxx()
{
    local path="$1"
    local display="$2"
    local tmp subs c
    tmp=$(mktemp "${TMPDIR:-/tmp}/av2p-migrate.XXXXXX")
    cp -- "$path" "$tmp"
    subs=0

    # Rule 1: include paths only, and only at the start of the included path.
    # Anchoring on '<' or '"' keeps already migrated aV2P paths untouched.
    c=$(count_include_matches "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E '/^[[:space:]]*#[[:space:]]*include/ s|([<"])alpaka/|\1aV2P/|g' "$tmp"
        subs=$((subs + c))
    fi

    # Rule 2: namespace.
    c=$(count_matches '\balpaka::' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|\balpaka::|aV2P::|g' "$tmp"
        subs=$((subs + c))
    fi

    # Rule 3: uppercase macros.
    c=$(count_matches '\bALPAKA_' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|\bALPAKA_|AV2P_|g' "$tmp"
        subs=$((subs + c))
    fi

    # Rule 4: lowercase macros.
    c=$(count_matches '\balpaka_' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|\balpaka_|aV2P_|g' "$tmp"
        subs=$((subs + c))
    fi

    report_device_lambdas "$tmp" "$display"
    finish_file "$path" "$display" "$tmp" "$subs"
}

process_cmake()
{
    local path="$1"
    local display="$2"
    local tmp subs c ln
    tmp=$(mktemp "${TMPDIR:-/tmp}/av2p-migrate.XXXXXX")
    cp -- "$path" "$tmp"
    subs=0

    # The native alpaka target-finalization prefix has no aV2P equivalent.
    while IFS= read -r ln; do
        [ -n "$ln" ] || continue
        warn_manual "${display}:${ln}" "ALPAKA_CMAKE_TARGET_ requires manual review"
    done < <(grep -n -o 'ALPAKA_CMAKE_TARGET_' "$tmp" 2>/dev/null | cut -d: -f1 || true)

    # Protect the literal prefix from the generic ALPAKA_ rename below.
    sed -i "s|ALPAKA_CMAKE_TARGET_|${CMAKE_SENTINEL}|g" "$tmp"

    # Rule 1: target pair before the generic target rule.
    c=$(count_matches '\balpaka::alpaka\b' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|\balpaka::alpaka\b|aV2P::aV2P|g' "$tmp"
        subs=$((subs + c))
    fi

    # Rule 2: any remaining target namespace.
    c=$(count_matches '\balpaka::' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|\balpaka::|aV2P::|g' "$tmp"
        subs=$((subs + c))
    fi

    # Rule 3: package lookup, keeping CONFIG/REQUIRED/version.
    c=$(count_matches 'find_package\([[:space:]]*alpaka\b' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|find_package\([[:space:]]*alpaka\b|find_package(aV2P|g' "$tmp"
        subs=$((subs + c))
    fi

    # Rule 4: uppercase option prefix becomes lowercase aV2P_, never AV2P_.
    c=$(count_matches '\bALPAKA_' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|\bALPAKA_|aV2P_|g' "$tmp"
        subs=$((subs + c))
    fi

    # Rule 5: lowercase CMake functions and variables.
    c=$(count_matches '\balpaka_' "$tmp")
    if [ "${c}" -gt 0 ]; then
        sed -i -E 's|\balpaka_|aV2P_|g' "$tmp"
        subs=$((subs + c))
    fi

    sed -i "s|${CMAKE_SENTINEL}|ALPAKA_CMAKE_TARGET_|g" "$tmp"

    finish_file "$path" "$display" "$tmp" "$subs"
}

process_root()
{
    local root="$1"
    local f rel display base skip pat
    while IFS= read -r -d '' f; do
        case "${root}" in
            .) rel="${f#./}" ;;
            *) rel="${f#"${root}"/}" ;;
        esac

        skip=0
        for pat in "${EXCLUDES[@]}"; do
            # shellcheck disable=SC2053
            if [[ "${rel}" == ${pat} ]]; then
                skip=1
                break
            fi
        done
        [ "${skip}" -eq 0 ] || continue

        if ! grep -Iq . "${f}" 2>/dev/null; then
            continue
        fi

        display="${f}"
        if [ "${display}" != "${display#./}" ]; then
            display="${display#./}"
        fi

        base="${f##*/}"
        case "${base}" in
            CMakeLists.txt|*.cmake|*.cmake.in)
                process_cmake "${f}" "${display}"
                ;;
            *)
                case "${base##*.}" in
                    hpp|h|hxx|hh|cpp|cxx|cc|cu|cuh|hip|ipp|inl|tpp)
                        process_cxx "${f}" "${display}"
                        ;;
                esac
                ;;
        esac
    done < <(find "${root}" \
        \( -type d \( \
            -name .git -o \
            -name .hg -o \
            -name .svn -o \
            -name build -o \
            -name 'build-*' -o \
            -name '_build*' -o \
            -name 'cmake-build-*' -o \
            -name _deps -o \
            -name third_party -o \
            -name thirdParty -o \
            -name third-party -o \
            -name vendor -o \
            -name external -o \
            -name node_modules -o \
            -name .cache -o \
            -name .venv \
        \) -prune \) \
        -o -type f -print0)
}

run_self_test()
{
    local dir rc out
    dir=$(mktemp -d "${TMPDIR:-/tmp}/av2p-migrate-selftest.XXXXXX")
    # shellcheck disable=SC2064
    trap "rm -rf '${dir}'" EXIT

    mkdir -p "${dir}/work" "${dir}/expected"

    cat > "${dir}/work/fixture.hpp" <<'EOF'
#include <alpaka/alpaka.hpp>
#include <myalpaka/thing.hpp>
#include "alpaka/vec/Vec.hpp"
#include <aV2P/alpaka/keep.hpp>
namespace alpaka { struct decoy {}; }
alpaka::Vec v;
ALPAKA_FN_ACC void f() {}
int myalpaka::x;
#define MY_ALPAKA_X 1
int xalpaka_y;
int alpaka_foo;
EOF

    cat > "${dir}/expected/fixture.hpp" <<'EOF'
#include <aV2P/alpaka.hpp>
#include <myalpaka/thing.hpp>
#include "aV2P/vec/Vec.hpp"
#include <aV2P/alpaka/keep.hpp>
namespace alpaka { struct decoy {}; }
aV2P::Vec v;
AV2P_FN_ACC void f() {}
int myalpaka::x;
#define MY_ALPAKA_X 1
int xalpaka_y;
int aV2P_foo;
EOF

    cat > "${dir}/work/CMakeLists.txt" <<'EOF'
find_package(alpaka 2.2 CONFIG REQUIRED)
target_link_libraries(app PRIVATE alpaka::alpaka)
alpaka_add_executable(app src/main.cpp)
set(ALPAKA_ACC_GPU_CUDA_ENABLE ON)
set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)
set(ALPAKA_CMAKE_TARGET_foo_bar ON)
EOF

    cat > "${dir}/expected/CMakeLists.txt" <<'EOF'
find_package(aV2P 2.2 CONFIG REQUIRED)
target_link_libraries(app PRIVATE aV2P::aV2P)
aV2P_add_executable(app src/main.cpp)
set(aV2P_ACC_GPU_CUDA_ENABLE ON)
set(aV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)
set(ALPAKA_CMAKE_TARGET_foo_bar ON)
EOF

    cp -- "${dir}/work/fixture.hpp" "${dir}/pre.hpp"

    set +e
    bash "${SELF}" --dry-run "${dir}/work" >/dev/null 2>&1
    rc=$?
    set -e
    if [ "${rc}" -ne 2 ]; then
        echo "FAIL: dry run exit status ${rc}, expected 2" >&2
        return 1
    fi
    if ! cmp -s -- "${dir}/pre.hpp" "${dir}/work/fixture.hpp"; then
        echo "FAIL: dry run modified a file" >&2
        return 1
    fi

    if ! bash "${SELF}" "${dir}/work" >/dev/null 2>&1; then
        echo "FAIL: migration run failed" >&2
        return 1
    fi

    if ! diff -u "${dir}/expected/fixture.hpp" "${dir}/work/fixture.hpp"; then
        echo "FAIL: C/C++ fixture mismatch" >&2
        return 1
    fi
    if ! diff -u "${dir}/expected/CMakeLists.txt" "${dir}/work/CMakeLists.txt"; then
        echo "FAIL: CMake fixture mismatch" >&2
        return 1
    fi
    if grep -q 'AV2P_' "${dir}/work/CMakeLists.txt"; then
        echo "FAIL: uppercase AV2P_ emitted in CMake output" >&2
        return 1
    fi

    cp -- "${dir}/work/fixture.hpp" "${dir}/first.hpp"
    cp -- "${dir}/work/CMakeLists.txt" "${dir}/first.cmake"

    out=$(bash "${SELF}" "${dir}/work" 2>/dev/null)
    case "${out}" in
        *"0 files changed, 0 substitutions"*) ;;
        *)
            echo "FAIL: second run was not a no-op" >&2
            echo "${out}" >&2
            return 1
            ;;
    esac
    if ! cmp -s -- "${dir}/first.hpp" "${dir}/work/fixture.hpp"; then
        echo "FAIL: second run changed the C/C++ fixture" >&2
        return 1
    fi
    if ! cmp -s -- "${dir}/first.cmake" "${dir}/work/CMakeLists.txt"; then
        echo "FAIL: second run changed the CMake fixture" >&2
        return 1
    fi

    echo "PASS: migration and idempotency self-test"
    return 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --exclude)
            [ $# -ge 2 ] || die "--exclude requires a GLOB"
            EXCLUDES+=("$2")
            shift 2
            ;;
        --exclude=*)
            EXCLUDES+=("${1#*=}")
            shift
            ;;
        --no-backup)
            shift
            ;;
        --quiet|-q)
            QUIET=1
            shift
            ;;
        --self-test)
            SELF_TEST=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do
                ROOTS+=("$1")
                shift
            done
            ;;
        -*)
            die "unknown option '$1'"
            ;;
        *)
            ROOTS+=("$1")
            shift
            ;;
    esac
done

if [ "${SELF_TEST}" -eq 1 ]; then
    run_self_test
    exit $?
fi

if [ ${#ROOTS[@]} -eq 0 ]; then
    ROOTS=(".")
fi

normalized=()
for root in "${ROOTS[@]}"; do
    root="${root%/}"
    [ -n "${root}" ] || root="/"
    [ -d "${root}" ] || die "not a directory: ${root}"
    normalized+=("${root}")
done
ROOTS=("${normalized[@]}")

for root in "${ROOTS[@]}"; do
    process_root "${root}"
done

if [ "${QUIET}" -eq 0 ]; then
    printf '%s files changed, %s substitutions\n' "${FILES_CHANGED}" "${TOTAL_SUBS}"
fi

if [ "${DRY_RUN}" -eq 1 ] && [ "${FILES_CHANGED}" -gt 0 ]; then
    exit 2
fi

exit 0
