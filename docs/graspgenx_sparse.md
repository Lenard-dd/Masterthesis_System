# Tabletop sparse grasp strategy

## Behavior

`config/grasping.yaml` selects `graspgenx_subprocess.strategy: sparse`.
The project server reuses the installed GraspGenX `run_graspmoe`, forces
`obb_density="sparse"`, and also returns diffusion candidates. It does not
switch automatically to dense sampling.

Default settings:

- 200 diffusion samples; 36 sparse planar angles; height offset `0` cm
  relative to the top face of the object point-cloud bounding box.
- Requested top-k is 100; the server preserves all OBB candidates for
  environment screening. The client applies the final sparse quota of 24
  after feasibility checks; the raw server count can exceed the requested top-k.
- Bounded MoveIt screening checks up to 30, reserving up to 12 for sparse.
- Both branches retain model scores and existing pose, score and motion checks.
- Sparse candidates use full orientation constraints to avoid introducing
  substantial tilt through relaxed planning.

The height offsets are sampling settings, not calibrated closing depths.
The current runner checks the open gripper mesh and closed-finger surface
against the environment cloud. For sparse candidates, constrained generation
searches backoff from 25 to 50 mm in 2.5 mm steps and checks target contact
support and the local support plane. These checks supplement tabletop-envelope
and MoveIt checks; they do not establish collision safety for unobserved geometry.
Inspect candidate clearance, contact depth and the final trajectory in RViz.

The old `-2,0` cm sampling setting is no longer used: the negative layer could
place grasps inside the tabletop for thin objects. The legacy single-pose
collision path remains available when constrained generation is disabled.

## Frames and limitations

Context-aware generation uses `TFPoseTransformer` to transform the object cloud
into `base_link` and saves a separate PLY, preserving the original camera PLY.
It updates the candidate frame, centroid and planar principal axis together.
The runner centers the new cloud; existing localization restores candidates to
`base_link`, followed by the configured grasp-frame-to-tool0 offset. Do not add
the centroid or tool length a second time.

Vertical means base-frame +Z. The implementation assumes the current tabletop
robot mounting; it does not rotate into the measured tabletop normal. A tilted
table needs an extended transform. TF uses the latest available transform, so
keep the arm stationary between cloud capture and generation. This is not
image-timestamp synchronization or hand-eye calibration.

Missing frame/TF or failed conversion reports a diffusion fallback. An older
server also reports fallback. Model errors are not treated as successful
inference and do not silently switch to mock observations.

## Startup

The GUI configuration points to the project server. Rebuild/source both packages
if using installed configuration, then restart the GUI and its GraspGenX server.
Already-running servers do not hot-reload code.

For an isolated manual diagnostic in the existing GraspGenX environment:

```bash
conda activate graspgenx
cd ~/GraspGenX/GraspGenX
python /home/lenard/dev_ws/src/LLM_Ros/scripts/graspgenx_sparse_server.py \
  --config ~/GraspGenX/GraspGenX/ext/graspgenx_checkpoints/release \
  --assets_dir ~/GraspGenX/GraspGenX/assets \
  --default_gripper robotiq_2f_140 \
  --port 5556
```

Adjust paths on another workstation. Run only one server on the port and retain
the GUI's SAM3/GraspGenX GPU handoff. Older diagnostic entry points without TF
context, such as the width test, continue to request diffusion.

## Validation and rollback

1. Start with GUI preview or `ros2 run llm_skill_robot ros-graspgenx-plan-only-demo`.
2. Check for `strategy=sparse`; investigate TF and server version on fallback.
3. Inspect `metadata.generation_branch` (`obb_sparse` or `diffusion`). Sparse
   approach axes should be close to base-frame `[0, 0, -1]`.
4. In RViz, check cloud/candidate alignment, jaw width, depth and orientation.
5. Record object dimensions, candidate branch/score, reachability and joint motion.

To roll back, set `graspgenx_subprocess.strategy: diffusion` and restart the
application reading that configuration. The extended server supports standard
`infer`, so it can remain in use, or use the upstream server.

Unit tests cover argument parsing, old-server fallback, branch quotas, cloud
rotation/centroid restoration and standard server actions. Mock tests do not
establish real model quality or physical grasp success.
