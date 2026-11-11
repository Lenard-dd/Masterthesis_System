# External dependencies and laboratory baseline

The primary workflow is supervised **real UR5 grasping and placement** through
`hitl_gui`. Fake-hardware simulation and mock tests remain available for
inspection and regression testing. These repositories contain the application
and its integration adapters; install the hardware stack and inference models
separately on the workstation.

## Call relationships

```text
hitl_gui component manager
  ├─ ur5_launcher/ur5_complete.launch.py
  │    ├─ UR driver/controllers and robot state publisher
  │    └─ MoveIt, combined UR5/Robotiq/camera model and lab calibration
  ├─ robotiq_gripper_communication/ros_gripper_comm.launch.py → ur_rtde
  └─ realsense2_camera/rs_launch.py → aligned RGB-D ROS topics

llm_skill_robot perception adapter
  → conda run -n sam3 → scripts/sam3_runner.py → installed SAM3 + checkpoint
  ← JSON results / mask files → localized object and environment point clouds

llm_skill_robot grasp adapter
  → conda run -n graspgenx → scripts/graspgenx_zmq_runner.py
  → ZMQ localhost:5556 → scripts/graspgenx_sparse_server.py
  → installed GraspGenX + model weights + gripper assets
  ← candidates → environment checks → MoveIt screening → GUI review/execution
```

The ROS interpreter does not import SAM3 or GraspGenX directly. Both model
packages are installed outside this Git checkout. The GUI releases SAM3 before
starting GraspGenX on the lab's 8 GB GPU. Retain that handoff.

## Hardware resources: obtain from the laboratory

The hardware setup was copied from the computer beside the UR5. Ask the lab
maintainer for that computer's matching source files, calibration and setup.
The inspected local copies are under `/home/lenard/dev_ws/src/ur_src/`:

| Path below `ur_src/` | Role |
|---|---|
| `ur5_launcher/` | Complete launch, MoveIt/controller settings, combined URDF/SRDF, mount meshes and robot calibration |
| `ros2_robotiq_gripper/robotiq_description/` | Robotiq 2F-140 model included by the combined URDF |
| `robotiq_gripper_communication/` | Actual lab gripper command/feedback node and services |

Copy complete package source directories, including `config/`, `launch/`,
`urdf/` and `meshes/` where present; do not copy only the top-level launch file.
Build them in the destination workspace. The combined model additionally uses
installed `ur_description`, `ur_moveit_config` and `realsense2_description`.
The old sibling `ur5_robotiq_description` and `ur5_robotiq_moveit_config` packages
are not the current GUI's hardware entry point.

The gripper package links to **ur_rtde 1.6.0**. The inspected build uses
`/home/lenard/dev_ws/install/ur_rtde_local/lib/cmake/ur_rtde` as `ur_rtde_DIR`.
Install the matching C++ development library and pass its actual CMake directory
when building that package on a new machine. Obtain the lab's build recipe;
this path is a local installation, not a source archive shipped here.

Both the GUI's UR launch arguments and the gripper's
`config/gripper_params.yaml` contain robot IP `192.168.10.27`. Update both for
another robot. Recheck `base_link`, `tool0`, `ur5_manipulator`, camera extrinsics,
recorded joint targets and tabletop calibration against the physical setup.

## Observed version baseline

Read from local Git revisions, environment metadata and installed ROS package
manifests on 2026-11-11. This is a reference inventory, not a fresh-install test
or a complete dependency lock.

| Component | Observed version |
|---|---|
| ROS | Humble; application Python 3.10 |
| NiceGUI | 3.15.0 |
| UR driver / UR MoveIt config | 2.14.0 / 2.14.0 |
| UR description | 2.13.0 |
| RealSense camera / description | 4.58.4 / 4.58.4 |
| MoveIt move_group | 2.5.10 |
| pymoveit2 | 4.2.0 |
| Robotiq description | 0.0.1 |
| ur_rtde | 1.6.0 |
| SAM3 environment | Python 3.12.13; torch 2.10.0+cu128; torchvision 0.25.0+cu128; NumPy 1.26.4 |
| GraspGenX environment | Python 3.10.20; torch 2.6.0+cu126; torchvision 0.21.0+cu126; NumPy 1.26.4; pyzmq 27.1.0 |

Source/asset revisions:

- SAM3: `8e451d5eb43c817b64ae7577fb7b9ae223db88a9` (`~/sam3`).
- SAM3 checkpoint: configured `facebook/sam3` snapshot
  `3c879f39826c281e95690f02c7821c4de09afae7`, file `sam3.pt`.
- GraspGenX: `53adf0499ad2b31391bc4b02cf51c9e715c452e1`
  (`~/GraspGenX/GraspGenX`).
- GraspGenX checkpoint repository: `7c834043c11a11417e31d6d5ea9355801e40a2c1`;
  `release/gen/epoch_736.pth` and `release/dis/epoch_1056.pth`.
- GraspGenX gripper descriptions: `8ed62d571ba4e3f46872983d331f5b1957b9fe74`.

The lab `ur5_launcher` and gripper communication packages report `0.0.0` and
have no independent Git revision in the inspected copies. Treat them as lab
working copies; record a source snapshot/checksum when handing them over.

## SAM3: installation and access

Sources: [official code and installation](https://github.com/facebookresearch/sam3),
[model access](https://huggingface.co/facebook/sam3).

Downloading code alone is insufficient. Install SAM3 and its dependencies in
an environment named `sam3`, obtain approved Hugging Face model access, and
download the checkpoint. The official prerequisites include Python >=3.12,
PyTorch >=2.7 and a CUDA-compatible GPU with CUDA >=12.6. Use the observed
revision/environment above as the reproduction baseline; do not silently
replace the configured SAM3 checkpoint with SAM 3.1.

The project runner imports `sam3.model_builder.build_sam3_image_model` and
`sam3.model.sam3_image_processor.Sam3Processor`. Installing only a Transformers
pipeline does not provide this integration. Point `sam3_subprocess.checkpoint_path`
in `config/perception.yaml` to the downloaded `sam3.pt` and ensure `conda` is
available to the GUI process. With a configured local checkpoint the runner
disables automatic Hugging Face model loading; an empty path requests it.
No SAM3 source or weights need to be copied into `LLM_Ros`.

## GraspGenX: installation and assets

Sources: [official code and installation](https://github.com/NVlabs/GraspGenX),
[model weights](https://huggingface.co/adithyamurali/GraspGenXModel),
[gripper assets used by the inspected revision](https://github.com/beininghan/gripper_descriptions).

Install the pinned source and dependencies into the `graspgenx` conda environment
(`pip install -e .` is supported by the inspected revision). Prepare both model
weights and gripper descriptions. Upstream supports fetching those on first
import, so initial provisioning requires network access; downloading the code
archive alone is insufficient. Confirm `robotiq_2f_140` assets are available.

Configure the checkpoint `release` directory, repository `assets` directory,
working directory and project server script in `hitl_gui/config/gui_config.yaml`.
The current GUI launches **this project's** `scripts/graspgenx_sparse_server.py`
inside that environment. It adds `infer_sparse` to the upstream ZMQ server.
An ordinary upstream server can cause a diffusion fallback rather than the
configured sparse behavior. See [the strategy guide](graspgenx_sparse.md).

The runner also imports GraspGenX collision/gripper utilities locally, so an
accessible server alone is not enough: the client-side `graspgenx` environment
must also have the matching package and assets installed. A service on another
computer would additionally require endpoint/process-management changes.

Upstream installation has evolved since the recorded revision. Its newer
native/TensorRT path is not required by this integration and is not validated
here. Prefer the recorded PyTorch baseline for reproducing the lab workflow.

## Destination workstation checklist

1. Install/build the lab hardware stack and source ROS Humble plus the workspace.
2. Install the GUI/backend dependencies using the [GUI guide](../../hitl_gui/README.md).
3. Install the two isolated model environments, weights and assets as above.
4. Replace workstation paths in GUI and backend configuration; set the API key
   in the GUI environment and use `ROS_DOMAIN_ID=27` in all ROS terminals.
5. Verify imports/model readiness and read-only robot diagnostics, then inspect
   localization and planned trajectories before supervised real execution.

A successful import or unit test does not replace model inference and physical
validation on the destination workstation. No model downloads, environment
installations or hardware launches were performed during this handoff cleanup.
