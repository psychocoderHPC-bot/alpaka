# pylint: disable=missing-docstring

"""Tests for the HIP GPU pinning before_script.

SPDX-License-Identifier: MPL-2.0
"""

import unittest

import bashi
import bashi.globals as g
from alpaka_bashi.globals import HWLOC, JOB_EXECUTION_RUNTIME, JOB_EXECUTION_TYPE
from alpaka_bashi.jobs_builder.default import construct_job_yaml

GPU_EXPORT = (
    "export ROCR_VISIBLE_DEVICES=$(( ${ALPAKA_CI_GPU_INDEX:-${CI_CONCURRENT_ID:-0}} % ${ALPAKA_CI_GPU_COUNT:-8} ))"
)


def get_combination(hip: bool) -> bashi.Combination:
    return bashi.parse_combination(
        [
            (g.HOST_COMPILER, g.CLANG, 18) if hip else (g.HOST_COMPILER, g.GCC, 13),
            (g.DEVICE_COMPILER, g.HIPCC, "6.2") if hip else (g.DEVICE_COMPILER, g.GCC, 13),
            (g.ALPAKA_ACC_GPU_HIP_ENABLE, g.ON if hip else g.OFF),
            (g.ALPAKA_ACC_GPU_CUDA_ENABLE, g.OFF),
            (g.ALPAKA_ACC_ONEAPI_CPU_ENABLE, g.OFF),
            (g.ALPAKA_ACC_ONEAPI_GPU_ENABLE, g.OFF),
            (g.ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, g.ON),
            (g.ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, g.OFF),
            (g.ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, g.OFF),
            (g.CMAKE, "3.30.3"),
            (g.UBUNTU, "24.04"),
            (JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME),
            (HWLOC, g.ON),
        ]
    )


class TestBeforeScript(unittest.TestCase):
    def build_job(self, hip: bool) -> dict:
        return construct_job_yaml(get_combination(hip), stage="", container_version="4.1", image_check=False)

    def test_hip_job_is_pinned_to_one_gpu(self):
        job = self.build_job(hip=True)
        self.assertEqual(job["before_script"], [GPU_EXPORT])
        self.assertEqual(job["variables"]["ALPAKA_CI_GPU_COUNT"], "8")

    def test_cpu_job_has_no_gpu_pinning(self):
        job = self.build_job(hip=False)
        self.assertFalse(job.get("before_script"))
        self.assertNotIn("ALPAKA_CI_GPU_COUNT", job["variables"])
