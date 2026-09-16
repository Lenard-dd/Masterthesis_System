#!/usr/bin/env python3
"""GraspGenX ZMQ server with an additive, top-down GraspMoE sparse action.

Run this in the GraspGenX environment from its repository root. The standard
health, metadata and infer actions retain their upstream behavior.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets_dir", required=True)
    parser.add_argument("--default_gripper", default="robotiq_2f_140")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    return parser.parse_args()


def main():
    args = parse_args()
    from graspgenx.samplers.graspmoe import run_graspmoe
    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    class SparseServer(GraspGenXZMQServer):
        def _handle_metadata(self):
            metadata = super()._handle_metadata()
            metadata["supported_actions"] = ["infer", "infer_sparse"]
            return metadata

        def _dispatch(self, request):
            if request.get("action") != "infer_sparse":
                return super()._dispatch(request)
            gripper_name = request.get("gripper_name") or self.default_gripper
            pc = np.asarray(request.get("point_cloud"), dtype=np.float32)
            if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) < 10:
                raise ValueError("infer_sparse requires an (N,3) point cloud with >=10 points")
            if not np.isfinite(pc).all():
                raise ValueError("infer_sparse point cloud contains non-finite values")
            if not gripper_name:
                raise ValueError("infer_sparse requires a gripper_name")
            num_grasps = int(request.get("num_grasps", 200))
            topk = int(request.get("topk_num_grasps", 100))
            sparse_quota = int(request.get("sparse_quota", 24))
            if num_grasps < 1 or topk < 1 or sparse_quota < 1:
                raise ValueError("num_grasps, topk_num_grasps and sparse_quota must be positive")
            sparse_quota = min(sparse_quota, topk)
            sampler = self._get_sampler(gripper_name)
            started = time.monotonic()
            result = run_graspmoe(
                pc, sampler,
                grasp_threshold=float(request.get("grasp_threshold", -1.0)),
                num_grasps=num_grasps,
                topk_num_grasps=-1,
                num_yaws=int(request.get("moe_num_yaws", 36)),
                z_offsets_cm=tuple(float(x) for x in request.get("moe_z_offsets_cm", [-2.0, 0.0])),
                obb_density="sparse",
            )
            obb_order = np.argsort(result["scores_obb"])[::-1][:sparse_quota]
            diff_order = np.argsort(result["scores_diff"])[::-1][:max(0, topk - len(obb_order))]
            grasps = np.concatenate(
                [result["grasps_obb"][obb_order], result["grasps_diff"][diff_order]], axis=0
            )
            scores = np.concatenate(
                [result["scores_obb"][obb_order], result["scores_diff"][diff_order]], axis=0
            )
            return {
                "grasps": grasps.astype(np.float32),
                "confidences": scores.astype(np.float32),
                "sources": ["obb_sparse"] * len(obb_order) + ["diffusion"] * len(diff_order),
                "gripper_name": gripper_name,
                "skipped_obb": bool(result["skipped_obb"]),
                "timing": {"infer_ms": (time.monotonic() - started) * 1000.0},
            }

    SparseServer(
        config_path=args.config, assets_dir=args.assets_dir,
        host=args.host, port=args.port, default_gripper=args.default_gripper,
    ).serve_forever()


if __name__ == "__main__":
    main()
