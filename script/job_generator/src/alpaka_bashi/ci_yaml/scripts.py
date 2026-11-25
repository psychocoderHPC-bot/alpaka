"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the script section of the GitLab CI test job yaml.
"""

from typing import Any

import bashi
from bashi.globals import ALPAKA_ACC_GPU_HIP_ENABLE, ON_VER
from typeguard import typechecked


@typechecked
def set_before_script(job_body: dict[str, Any], combination: bashi.Combination):
    """Set a before_script which pins every HIP job to one GPU.

    The device index is the runner concurrency slot modulo the number of GPUs, so concurrent jobs
    on the same runner use different devices. ALPAKA_CI_GPU_COUNT can be overridden for runners
    with a different GPU count and ALPAKA_CI_GPU_INDEX for a single job. When concurrency exceeds
    ALPAKA_CI_GPU_COUNT, the indices wrap and jobs share devices again, so ALPAKA_CI_GPU_COUNT must
    match the runner's GPU count.

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
    if combination[ALPAKA_ACC_GPU_HIP_ENABLE].version == ON_VER:
        job_body["before_script"] = [
            "export ROCR_VISIBLE_DEVICES=$(( ${ALPAKA_CI_GPU_INDEX:-${CI_CONCURRENT_ID:-0}} % ${ALPAKA_CI_GPU_COUNT:-8} ))"
        ]


@typechecked
def set_script(job_body: dict[str, Any]):
    """Set the job section of a job. Overwrite an existing job section."""
    job_body["script"] = [
        "$APCI_ALPAKA_ROOT/script/ci/info.sh",
        "$APCI_ALPAKA_ROOT/script/ci/install.sh",
        "$APCI_ALPAKA_ROOT/script/ci/configure.sh",
        "$APCI_ALPAKA_ROOT/script/ci/build.sh",
        "$APCI_ALPAKA_ROOT/script/ci/test.sh",
    ]
