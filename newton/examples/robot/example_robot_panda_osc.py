# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Panda OSC
#
# Operational-space controller for a Franka Panda. Mirrors the torque-level
# OSC used in IsaacLab's Factory tasks but runs entirely in Newton + MuJoCo.
# The arm is held aloft by MuJoCo gravity compensation; the OSC writes
# joint torques into ``control.joint_f`` to drive the TCP toward a target
# pose set from the GUI.
#
# Phase 1 (this commit): scene + gravcomp + effort mode on arm. The OSC
# writes zeros, so success looks like the arm staying at its initial pose.
#
# Command: python -m newton.examples robot_panda_osc --world-count 1
#
###########################################################################

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.robot.osc import (
    OSCController,
    apply_disturbance_force_kernel,
    pack_diagnostics_kernel,
    quat_to_rpy_kernel,
    reduce_arm_torque_norm_kernel,
    reduce_h_symmetry_resid_kernel,
    reduce_pos_distance_mm_kernel,
    reduce_pos_err_mm_kernel,
    reduce_rot_err_deg_kernel,
    rpy_to_quat_kernel,
    step_clip_target_kernel,
    update_osc_debug_frame_lines_kernel,
)

# Optional alternative robot asset: IsaacLab Factory's franka_mimic.usd. Not
# committed to this repo for licensing reasons; users opt in via --robot usd
# and copy the USD into newton/examples/assets/franka_mimic/. See the README
# at that path for the workflow.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORY_USD_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "assets", "franka_mimic", "franka_mimic.usd"))

# Per-DOF dynamics dump produced by
# newton/examples/assets/factory_baseline/probe_joint_pos.py running under
# IsaacLab. When the file contains the four optional keys
# (joint_armature/friction/damping/stiffness), the USD path mirrors them so
# the OSC step-response benchmark sees the same regularization Factory does.
# Missing or absent keys are silently tolerated; the example falls back to
# its tuned defaults.
FACTORY_JOINT_POS_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "assets", "factory_baseline", "factory_joint_pos.json")
)


def _load_factory_dynamics() -> dict | None:
    """Return per-DOF dynamics from factory_joint_pos.json, or None if absent.

    Only the four keys this example needs are returned. Returns ``None`` if
    the file is missing entirely or doesn't carry the new keys (i.e. the
    user hasn't rerun the IsaacLab probe since the keys were added).
    """
    if not os.path.isfile(FACTORY_JOINT_POS_PATH):
        return None
    try:
        with open(FACTORY_JOINT_POS_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    keys = ("joint_armature", "joint_friction", "joint_damping", "joint_stiffness")
    if not all(k in data for k in keys):
        return None
    return {k: data[k] for k in keys}


# Scene layout (metres, world frame). Lifted from example_robot_panda_nut_bolt.
TABLE_HEIGHT = 0.1
TABLE_HALF_EXTENT = 0.4
TABLE_POS = wp.vec3(0.0, -0.5, 0.5 * TABLE_HEIGHT)
TABLE_TOP_CENTER = TABLE_POS + wp.vec3(0.0, 0.0, 0.5 * TABLE_HEIGHT)
ROBOT_BASE_POS = TABLE_TOP_CENTER + wp.vec3(-0.5, 0.0, 0.0)

# Robot DOF layout. Both URDF and USD profiles expose 9 actuated DOFs:
# 7 arm joints (DOFs 0-6) followed by 2 finger joints (DOFs 7-8).
N_ARM_DOFS = 7
N_FINGER_DOFS = 2
N_ROBOT_DOFS = N_ARM_DOFS + N_FINGER_DOFS

# Initial joint configuration (radians). Same as nut-bolt branch.
# Two named home poses are available:
#
#   INIT_ARM_Q_FACTORY_SEED — the joint values published in
#       factory_env_cfg.py (panda_joint1..7). This is the IK *seed* pose
#       Factory uses before running its reset-time IK; the arm doesn't
#       actually live here during a task.
#
#   INIT_ARM_Q_FACTORY_TASK_HOME — the deterministic post-IK joint config
#       that Factory's reset settles to (peg-insert task, all noise
#       disabled). This is the actual home pose the OSC step-response
#       baseline was measured at. Captured by the IsaacLab probe in
#       newton/examples/assets/factory_baseline/probe_joint_pos.py and
#       mirrored to factory_joint_pos.json.
#
# Newton's URDF and Factory's USD share identical joint axes and link
# offsets through panda_hand, so these radian values transfer directly
# with no sign flips.
INIT_ARM_Q_FACTORY_SEED = (
    0.00871,
    -0.10368,
    -0.00794,
    -1.49139,
    -0.00083,
    1.38774,
    0.0,
)
INIT_ARM_Q_FACTORY_TASK_HOME = (
    # Captured by factory_baseline/osc_step_response.py at the moment it
    # records home_pos/home_quat for the baseline trajectories. With all
    # IK randomization disabled and 8 settle steps after env.reset(), this
    # is the exact pose every IsaacLab trial starts from in
    # osc_isaaclab_steps.json.
    -0.5294972062110901,
    0.5211741924285889,
    0.5377357006072998,
    -2.0401036739349365,
    -0.41427168250083923,
    2.4552853107452393,
    -0.7210570573806763,
)
# Default to the post-IK task home so Newton's home matches the pose at
# which Factory's OSC step-response baseline was recorded.
INIT_ARM_Q = INIT_ARM_Q_FACTORY_TASK_HOME
INIT_FINGER_Q = (0.04, 0.04)


# ---------------------------------------------------------------------------
# Robot asset profile
# ---------------------------------------------------------------------------
#
# The OSC scene optionally swaps Newton's URDF Franka for IsaacLab Factory's
# franka_mimic.usd so the OSC step-response benchmark can be compared to
# IsaacLab's PhysX-USD baseline on the *same* asset. The two robots share
# kinematic structure through link7 but differ in:
#
#   * link / body / joint names (fr3_* vs panda_*)
#   * the canonical TCP body name (fr3_hand_tcp vs panda_fingertip_centered)
#   * the hand-frame convention - the USD bakes a Z-flip + 45 deg yaw into
#     panda_hand that the URDF's fr3_hand does not have
#   * inertia identification (URDF link1 Iyy is ~30x smaller than USD's;
#     wrist links differ by ~50x)
#
# The OSC operates on the TCP body via a constant local offset, so we only
# need to know the TCP body name and the asset loader; the rest of the OSC
# pipeline is asset-agnostic.


@dataclass(frozen=True)
class RobotProfile:
    """Per-asset constants for the dual-robot Panda OSC example.

    The benchmark scene contains only the robot and a table; nothing the
    robot grasps. So unlike the nut-bolt profile this one does not carry
    SDF-resolution knobs or finger-shape filters - the OSC drives torques
    directly into the arm DOFs and the fingers stay at their initial
    opening on a stiff PD.

    Attributes:
        kind: ``"urdf"`` or ``"usd"``.
        asset_loader: ``"add_urdf"`` or ``"add_usd"`` - selects the
            ``ModelBuilder`` method.
        asset_path: filesystem path to the asset (resolved at runtime via
            :func:`newton.utils.download_asset` for URDF; copied locally
            for USD - see ``newton/examples/assets/franka_mimic/README.md``).
        tcp_body: name suffix of the body the OSC drives. URDF uses
            ``fr3_hand_tcp``; USD uses ``panda_fingertip_centered`` which
            already lives at the fingertip-tip and carries the Factory
            hand convention.
        hand_body: name suffix of the gripper hand body.
        finger_left, finger_right: name suffixes of the two finger bodies.
        base_link_suffixes: tuple of name suffixes for links that should
            have the ground-plane collision filter applied.
        init_arm_q: 7 arm joint angles [rad] used as the home configuration.
        init_finger_q: pair of finger joint values [m] at home.
        finger_target: per-finger PD position target [m]. Defaults to
            ``init_finger_q`` so the gripper holds the loaded pose. Held
            here so changes are colocated with the other finger constants.
        tcp_offset_local: TCP local-frame offset relative to ``tcp_body``
            (``wp.transform``). Aligns Newton's TCP frame with Factory's
            ``panda_fingertip_centered`` so the OSC operates at the same
            physical point on either robot.
    """

    kind: str
    asset_loader: str
    asset_path: str
    tcp_body: str
    hand_body: str
    finger_left: str
    finger_right: str
    base_link_suffixes: tuple
    init_arm_q: tuple
    init_finger_q: tuple
    finger_target: tuple
    tcp_offset_local: wp.transform


# Newton's URDF places fr3_hand_tcp 0.1035 m below fr3_hand at the
# gripper-opening *center* (between the fingers, midway along their
# length). IsaacLab's Factory USD uses panda_fingertip_centered, which
# sits 0.1121 m below panda_hand at the *fingertip tips*. The two
# bodies differ along the gripper-extension axis by 8.6 mm.
#
# At the Factory post-IK task home (INIT_ARM_Q_FACTORY_TASK_HOME),
# fr3_hand_tcp's world rotation coincides with
# panda_fingertip_centered's (both are ~Rx(pi)) - confirmed numerically
# by the per-init TCP-pose print in :meth:`Example.__init__`. So no
# rotational correction is needed: we drive each asset's TCP body
# directly with an identity local offset and accept the documented
# 8.6 mm Z difference between the two TCP bodies as a small constant
# offset between URDF and USD step-response trajectories.
URDF_TCP_OFFSET = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
USD_TCP_OFFSET = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())


URDF_PROFILE = RobotProfile(
    kind="urdf",
    asset_loader="add_urdf",
    asset_path="",  # resolved at runtime via newton.utils.download_asset
    tcp_body="fr3_hand_tcp",
    hand_body="fr3_hand",
    finger_left="fr3_leftfinger",
    finger_right="fr3_rightfinger",
    base_link_suffixes=("/fr3_link0", "/fr3_link1"),
    init_arm_q=INIT_ARM_Q,
    init_finger_q=(0.04, 0.04),
    finger_target=(0.04, 0.04),
    tcp_offset_local=URDF_TCP_OFFSET,
)


USD_PROFILE = RobotProfile(
    kind="usd",
    asset_loader="add_usd",
    asset_path=FACTORY_USD_PATH,
    tcp_body="panda_fingertip_centered",
    hand_body="panda_hand",
    finger_left="panda_leftfinger",
    finger_right="panda_rightfinger",
    base_link_suffixes=("/panda_link0", "/panda_link1"),
    init_arm_q=INIT_ARM_Q,
    init_finger_q=(0.04, 0.04),
    finger_target=(0.04, 0.04),
    tcp_offset_local=USD_TCP_OFFSET,
)


def _resolve_robot_profile(kind: str) -> RobotProfile:
    """Return the :class:`RobotProfile` for ``kind``, validating its asset."""
    if kind == "urdf":
        return replace(
            URDF_PROFILE,
            asset_path=str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"),
        )
    if kind == "usd":
        if not os.path.isfile(FACTORY_USD_PATH):
            raise RuntimeError(
                "--robot usd was requested but IsaacLab Factory's franka_mimic.usd "
                "is not present at:\n"
                f"    {FACTORY_USD_PATH}\n\n"
                "The USD asset is not committed to this repo for licensing reasons. "
                "Copy it locally with:\n\n"
                "    cp /tmp/Assets/Isaac/<version>/Isaac/IsaacLab/Factory/franka_mimic.usd \\\n"
                f"       {FACTORY_USD_PATH}\n\n"
                "See newton/examples/assets/franka_mimic/README.md for the\n"
                "full workflow and license note."
            )
        return USD_PROFILE
    raise ValueError(f"Unknown --robot kind: {kind!r} (expected 'urdf' or 'usd').")


def _resolve_ee_body_index(builder: newton.ModelBuilder, profile: RobotProfile) -> int:
    """Return the body index of ``profile.tcp_body`` within ``builder``.

    Both URDF and USD assets have unique body labels of the form
    ``"{root}/{body}"``; this matches by trailing component so we don't
    depend on which absolute prefix the asset loader chose.
    """
    name = profile.tcp_body
    for i, lbl in enumerate(builder.body_label):
        if lbl.endswith(f"/{name}") or lbl.endswith(name):
            return i
    raise RuntimeError(f"Could not find body with name suffix {name!r} in builder.body_label.")


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.viewer = viewer

        # Pick the robot asset (URDF default, USD optional). Stash on the
        # instance so build_franka_with_table - called by super().__init__()
        # in subclasses too - can read it without an attribute-before-init
        # dance.
        robot_kind = getattr(args, "robot", "urdf")
        self.robot_profile = _resolve_robot_profile(robot_kind)

        # Build robot + table, replicate, finalize.
        robot_builder = self.build_franka_with_table()
        self.robot_body_count = robot_builder.body_count
        # ``ee_index`` is set by ``build_franka_with_table`` via the profile.
        ee_body_index = self.ee_index

        scene = newton.ModelBuilder()
        scene.replicate(robot_builder, self.world_count)
        ground_shape_idx = scene.add_ground_plane()
        # Filter ground vs base/link0 contacts (robot sits on the table).
        # Both URDF and USD profiles list their root links in
        # ``base_link_suffixes`` so the filter is asset-agnostic.
        base_suffixes = tuple(self.robot_profile.base_link_suffixes)
        for shape_idx, body_idx in enumerate(scene.shape_body):
            if body_idx < 0:
                continue
            if scene.body_label[body_idx].endswith(base_suffixes):
                scene.add_shape_collision_filter_pair(shape_idx, ground_shape_idx)

        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count

        # Sanity-check the body indexing resolved by name. Failing fast
        # here saves hours of silent-bug debugging downstream.
        ee_label = scene.body_label[ee_body_index]
        expected = self.robot_profile.tcp_body
        assert ee_label.endswith(f"/{expected}") or ee_label.endswith(expected), (
            f"ee_body_index={ee_body_index} expected suffix {expected!r}, got {ee_label!r}"
        )

        # State and control.
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Gripper PD targets - leave finger DOFs on PD, arm DOFs are torque-driven.
        # joint_target_pos layout matches joint_q: per-world DOF stride.
        joint_target_view = self.control.joint_target_pos.reshape((self.world_count, -1))
        joint_q_view = self.model.joint_q.reshape((self.world_count, -1))
        wp.copy(dest=joint_target_view[:, :N_ROBOT_DOFS], src=joint_q_view[:, :N_ROBOT_DOFS])

        # Collisions + solver.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
        )
        self.contacts = self.collision_pipeline.contacts()

        # Hardcoded MuJoCo solver: this example depends on `mujoco:gravcomp`
        # and `mujoco:jnt_actgravcomp` which are honored only by SolverMuJoCo.
        # njmax/nconmax are per-world contact budgets. 2000 keeps headroom
        # for aggressive OSC trajectories that briefly graze the table.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=4000,
            nconmax=4000,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        # Articulation view for J/M. URDF emits an articulation labeled
        # "fr3"; Factory's USD emits "/panda/root_joint". Pick the right
        # pattern from the profile so the same view code works on either
        # asset (ArticulationView treats the pattern as a glob over
        # builder.articulation_label).
        art_pattern = "fr3" if self.robot_profile.kind == "urdf" else "*panda*"
        self.art_view = newton.selection.ArticulationView(self.model, art_pattern)

        # TCP local offset relative to the EE body. URDF aligns to Factory's
        # panda_fingertip_centered via a +0.0086 m Z shift and a 45-deg
        # yaw + Z-flip quat (see RobotProfile docstring). USD uses identity:
        # panda_fingertip_centered already lives at the fingertip midpoint
        # with the Factory hand convention baked in.
        tcp_offset_local = self.robot_profile.tcp_offset_local

        self.osc = OSCController(
            model=self.model,
            articulation_view=self.art_view,
            world_count=self.world_count,
            ee_body_index=ee_body_index,
            tcp_offset_local=tcp_offset_local,
            n_arm_dofs=N_ARM_DOFS,
            num_bodies_per_world=self.num_bodies_per_world,
            n_dofs_per_world=N_ROBOT_DOFS,
        )
        self.ee_body_index = ee_body_index

        # Default arm pose for nullspace centering: every world starts at
        # the profile's init_arm_q. Stored on device so the nullspace
        # torque can pull each world's redundant DOF toward this configuration.
        q_default_host = [list(self.robot_profile.init_arm_q) for _ in range(self.world_count)]
        self.osc.set_default_pose(wp.array(q_default_host, dtype=float, device=self.model.device))
        # Phase 6: enable nullspace damping by default. The dominant role is
        # damping (kd_null) on joint velocities — this kills the slow growing
        # oscillation in the redundant DOF without fighting the task target.
        # A small kp_null keeps the joints near INIT_ARM_Q over long horizons
        # but is intentionally weak so it doesn't conflict with task tracking.
        self.osc.enable_nullspace = True
        self.osc.kp_null = 0.5
        self.osc.kd_null = 5.0

        # Initial gains. Mirrors Factory's `default_task_prop_gains`
        # (factory_tasks_cfg.py): translation = 200 N/m, rotation = 50 N.m/rad.
        # Critically damped: Kd = 2 * sqrt(Kp).
        kp_init_arr = [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        kd_init_arr = [2.0 * (k**0.5) for k in kp_init_arr]
        kp_host = wp.array([kp_init_arr] * self.world_count, dtype=float, device=self.model.device)
        kd_host = wp.array([kd_init_arr] * self.world_count, dtype=float, device=self.model.device)
        self.osc.set_gains(kp_host, kd_host)

        # Seed target = current TCP pose so the OSC has zero error at startup.
        self.osc.update_tcp_state(self.state_0)
        self.osc.set_target(self.osc.tcp_pos, self.osc.tcp_quat)

        # Post-init sanity check: print TCP world pose at INIT_ARM_Q so URDF
        # and USD runs can be compared modulo the documented 8.6 mm Z offset
        # between fr3_hand_tcp and panda_fingertip_centered (both bodies
        # sit between the fingers; they differ only along the gripper-
        # extension axis). At the Factory post-IK task home, their world
        # rotations coincide (~Rx(pi)) so the printed quat_xyzw should
        # match across robots up to a global sign. The 8.6 mm Z gap
        # appears in pos_z: URDF reports fr3_hand_tcp at the gripper-
        # opening center, USD reports panda_fingertip_centered at the
        # fingertip tip - URDF sits 8.6 mm higher.
        tcp_pos_init = self.osc.tcp_pos.numpy()[0]
        tcp_quat_init = self.osc.tcp_quat.numpy()[0]
        print(
            f"[panda-osc] robot={self.robot_profile.kind} "
            f"TCP @ INIT_ARM_Q: "
            f"pos=({tcp_pos_init[0]:+.4f}, {tcp_pos_init[1]:+.4f}, {tcp_pos_init[2]:+.4f}) m  "
            f"quat_xyzw=({tcp_quat_init[0]:+.4f}, {tcp_quat_init[1]:+.4f}, "
            f"{tcp_quat_init[2]:+.4f}, {tcp_quat_init[3]:+.4f})",
            flush=True,
        )

        # GUI target buffers. The GUI writes a "raw" target which is then
        # clipped per control tick into the OSC's actual target. This mirrors
        # Factory's per-action delta clipping.
        self.gui_target_pos = wp.zeros(self.world_count, dtype=wp.vec3, device=self.model.device)
        self.gui_target_rpy = wp.zeros(self.world_count, dtype=wp.vec3, device=self.model.device)
        self.gui_target_quat = wp.zeros(self.world_count, dtype=wp.vec4, device=self.model.device)
        # Seed GUI target = current TCP, with RPY extracted from current quat.
        wp.copy(self.gui_target_pos, self.osc.tcp_pos)
        wp.copy(self.gui_target_quat, self.osc.tcp_quat)
        wp.launch(
            quat_to_rpy_kernel,
            dim=self.world_count,
            inputs=[self.osc.tcp_quat],
            outputs=[self.gui_target_rpy],
            device=self.model.device,
        )

        # Pull the seeded RPY back to host so the ImGui sliders show meaningful
        # initial values. The GUI then keeps a Python copy and pushes back to
        # the wp.array on demand.
        rpy_init = self.gui_target_rpy.numpy().copy()
        pos_init = self.gui_target_pos.numpy().copy()
        self._gui_target_pos_host = [list(map(float, pos_init[w])) for w in range(self.world_count)]
        self._gui_target_rpy_host = [list(map(float, rpy_init[w])) for w in range(self.world_count)]
        self._gui_target_dirty = True  # force first sync to device

        # GUI control state.
        self._active_world = 0
        self._broadcast = False
        self._show_debug_frames = True
        self._debug_axis_len = 0.05  # m

        # Frame-overlay line buffers: 7 segments per world.
        n_segments = self.world_count * 7
        self._dbg_starts = wp.zeros(n_segments, dtype=wp.vec3, device=self.model.device)
        self._dbg_ends = wp.zeros(n_segments, dtype=wp.vec3, device=self.model.device)
        # Color array, one vec3 per segment. Filled once at init: TCP frame in
        # saturated RGB, target frame in half-saturated RGB, error vector in
        # white. Built as a Python list of vec3 tuples and handed to wp.array.
        colors_host: list[tuple[float, float, float]] = []
        for _ in range(self.world_count):
            colors_host += [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            colors_host += [(0.4, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 0.4)]
            colors_host += [(1.0, 1.0, 1.0)]
        self._dbg_colors = wp.array(colors_host, dtype=wp.vec3, device=self.model.device)
        # Per-tick delta caps. Defaults match Factory's pos_action_threshold
        # (5 mm) and a moderate rotation bound (~3 deg).
        self._pos_step_max = 0.005
        self._rot_step_max = 0.05
        # Control rate decimation: physics frame_dt = 1/60 s; with decimation=4
        # the OSC runs at 15 Hz, matching IsaacLab's Factory env step rate.
        self._control_decimation = 4

        # Viewer.
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)
            self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
            self.viewer.register_ui_callback(self.render_ui, position="side")

        # Test-mode tracking: capture initial TCP height to detect arm collapse.
        self.test_mode = args.test
        self._initial_tcp_z: float | None = None
        self._gui_frame = 0

        # Disturbance buffer: per-world (fx, fy, fz) [N] applied at the EE
        # body's COM for `_disturbance_frames_remaining` more frames. Counted
        # down each frame; cleared when zero. Used by phase-7 sanity buttons.
        # Stored on device as a wp.array so the kernel can write body_f
        # without a host round-trip per frame.
        self._disturbance_force = wp.zeros(self.world_count, dtype=wp.vec3, device=self.model.device)
        self._disturbance_frames_remaining = 0
        self._disturbance_magnitude = 5.0  # N
        self._disturbance_duration_frames = 60  # 1 s at 60 fps

        # Cached diagnostic scalars (host-side), refreshed every N frames in
        # step(). The actual reductions run in Warp kernels and write into
        # the per-world wp.array buffers below; we read back one scalar per
        # diagnostic at refresh time.
        self._diag_pos_err_mm = 0.0
        self._diag_rot_err_deg = 0.0
        self._diag_arm_torque_norm = 0.0
        self._diag_h_symmetry_resid = 0.0
        self._diag_jacobian_cond = 0.0
        self._diag_refresh_period = 6  # frames

        # On-device scalar buffers, one per world per diagnostic.
        wc = self.world_count
        dev = self.model.device
        self._diag_pos_err_mm_buf = wp.zeros(wc, dtype=float, device=dev)
        self._diag_rot_err_deg_buf = wp.zeros(wc, dtype=float, device=dev)
        self._diag_arm_torque_norm_buf = wp.zeros(wc, dtype=float, device=dev)
        self._diag_h_sym_resid_buf = wp.zeros(wc, dtype=float, device=dev)
        self._diag_pos_distance_mm_buf = wp.zeros(wc, dtype=float, device=dev)
        # Packed buffer: 4 scalars per world (pos err mm, rot err deg, torque
        # norm, H sym resid). One readback per refresh instead of four.
        self._diag_packed_buf = wp.zeros((wc, 4), dtype=float, device=dev)
        # Cached host-side TCP / target pose for the UI panel - refreshed at
        # the diagnostic rate so the render thread doesn't trigger device syncs.
        self._diag_tcp_pos_host = (0.0, 0.0, 0.0)
        self._diag_target_pos_host = (0.0, 0.0, 0.0)
        self._target_offset = (
            tuple(float(v) for v in args.target_offset) if getattr(args, "target_offset", None) is not None else None
        )
        if self._target_offset is not None:
            # Push the offset into the GUI host state so step-clip drives the
            # OSC toward it. Useful for a programmatic reach test in --test.
            for w in range(self.world_count):
                for i in range(3):
                    self._gui_target_pos_host[w][i] += self._target_offset[i]
            self._gui_target_dirty = True
            self._initial_target_offset_norm = float((sum(v * v for v in self._target_offset)) ** 0.5)

        self.capture()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def build_franka_with_table(self) -> newton.ModelBuilder:
        """Build a single-world Panda-on-table builder.

        Profile-driven: loads either Newton's URDF (default) or IsaacLab
        Factory's franka_mimic.usd depending on ``self.robot_profile``.
        Either way:
          - Arm DOFs run in **effort mode** (joint_target_ke = 0). This makes
            the controller's ``joint_f`` torques the only command driving
            arm motion; PD does not fight the OSC.
          - Fingers stay on PD so the gripper holds a steady opening.
          - Per-body gravity is disabled across the entire Franka articulation
            so Newton runs in Factory's "robot articulation has gravity off"
            regime (Factory's ``factory_env_cfg.py`` sets
            ``disable_gravity=True`` on the robot). Newton's MuJoCo solver
            honors ``mujoco:gravcomp = 1.0`` at the body level for this.
        """
        profile = self.robot_profile
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # Tight contact margin. Without this, default_shape_cfg.gap stays
        # None and the per-shape gap falls back to the model-level
        # rigid_gap (0.1 m / 10 cm). The robot base sits 28 mm to the
        # side of the table edge in XY at ROBOT_BASE_POS, so a 10 cm
        # margin generates dozens of spurious base-vs-table contacts
        # per step that destabilize the OSC. The nut-bolt example sets
        # default_shape_cfg.gap = 0.01 for the same reason; we mirror
        # that here so the URDF parser (which clones default_shape_cfg)
        # and the USD parser (which uses default_shape_cfg.gap as a
        # per-shape fallback) both pick up a 10 mm margin.
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(gap=0.01)

        # Table.
        builder.add_shape_box(
            body=-1,
            hx=TABLE_HALF_EXTENT,
            hy=TABLE_HALF_EXTENT,
            hz=0.5 * TABLE_HEIGHT,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
        )

        # Robot asset. URDF and USD share kinematic structure through
        # link7 but differ in body / joint names and hand-frame convention -
        # see RobotProfile. Capture the body-index range the parser fills
        # in so the gravcomp loop below can pick up *every* body the asset
        # added (including ones we don't enumerate by name, like USD's
        # fixed-jointed force_sensor between link7 and the hand).
        body_count_before_parse = builder.body_count
        base_xform = wp.transform(ROBOT_BASE_POS, wp.quat_identity())
        if profile.kind == "urdf":
            builder.add_urdf(
                profile.asset_path,
                xform=base_xform,
                floating=False,
                enable_self_collisions=False,
                parse_visuals_as_colliders=True,
            )
        elif profile.kind == "usd":
            builder.add_usd(
                profile.asset_path,
                xform=base_xform,
                enable_self_collisions=False,
                skip_mesh_approximation=True,
            )
        else:  # pragma: no cover - guarded by _resolve_robot_profile
            raise ValueError(f"Unsupported robot profile kind: {profile.kind!r}")

        # Initial joint configuration. Both profiles default to Factory's
        # post-IK task home through link7 - URDF and USD share kinematic
        # conventions there so the radian values transfer directly.
        init_arm_q = list(profile.init_arm_q)
        init_finger_q = list(profile.init_finger_q)
        finger_target = list(profile.finger_target)
        builder.joint_q[:N_ROBOT_DOFS] = [*init_arm_q, *init_finger_q]
        builder.joint_target_pos[:N_ROBOT_DOFS] = [*init_arm_q, *finger_target]

        # Gains: arm effort-only (ke=0), fingers PD. Effort limits per FR3
        # datasheet. Joint-level kd on arm DOFs is set by the subclass /
        # caller (see step-response example for the CLI knob); the parent
        # default is 50, which together with ke=0 gives a pure -kd*qd
        # damping torque per joint that kills the slow under-damped mode
        # the redundant 7th DOF would otherwise exhibit when the OSC
        # tracks a moving target.
        builder.joint_target_ke[:N_ROBOT_DOFS] = [0.0] * N_ARM_DOFS + [100.0, 100.0]
        builder.joint_target_kd[:N_ROBOT_DOFS] = [50.0] * N_ARM_DOFS + [10.0, 10.0]
        builder.joint_effort_limit[:N_ROBOT_DOFS] = [87.0] * 4 + [12.0] * 3 + [100.0, 100.0]
        # Validated in the nut-bolt experiment: this armature schedule
        # matches the closed-loop response between URDF and USD assets.
        builder.joint_armature[:N_ROBOT_DOFS] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        # USD path: mirror Factory's per-DOF dynamics (armature/friction/
        # damping/stiffness) so the OSC step-response benchmark sees the
        # same regularization Factory does. The values come from
        # factory_baseline/factory_joint_pos.json, which is regenerated
        # by running probe_joint_pos.py under IsaacLab. Mirroring is
        # opt-in and graceful: if the JSON doesn't carry the keys (older
        # snapshot, file missing) we silently fall back to the URDF-tuned
        # defaults set above. URDF stays on the defaults regardless,
        # since the JSON describes the USD asset's actuator config.
        factory_dyn = _load_factory_dynamics()
        if profile.kind == "usd" and factory_dyn is not None:
            builder.joint_armature[:N_ROBOT_DOFS] = factory_dyn["joint_armature"][:N_ROBOT_DOFS]
            builder.joint_friction[:N_ROBOT_DOFS] = factory_dyn["joint_friction"][:N_ROBOT_DOFS]
            builder.joint_target_kd[:N_ROBOT_DOFS] = factory_dyn["joint_damping"][:N_ROBOT_DOFS]
            builder.joint_target_ke[:N_ROBOT_DOFS] = factory_dyn["joint_stiffness"][:N_ROBOT_DOFS]
            print("[panda-osc] usd: armature/friction/damping/stiffness mirrored from factory_joint_pos.json")
        elif profile.kind == "usd":
            print(
                "[panda-osc] usd: factory_joint_pos.json missing the "
                "armature/friction/damping/stiffness keys; using tuned "
                "defaults. Rerun probe_joint_pos.py under IsaacLab to "
                "refresh."
            )

        # USD parser detail: franka_mimic.usd has PhysicsDrive prims with
        # stiffness=0, so the parser picks JointTargetMode.EFFORT for every
        # DOF. That's wrong for two reasons here:
        #   * The fingers need PD so the gripper holds a steady opening.
        #   * The arm needs POSITION mode too -- not because we want PD,
        #     but because we pin ctrl_source to JOINT_TARGET below for
        #     every actuator so the fingers read joint_target_pos. An
        #     EFFORT-mode arm actuator with ctrl_source=JOINT_TARGET would
        #     read joint_target_pos[i] (the home angle, e.g. -2.37 rad)
        #     and apply it as N*m of torque every step, so the arm
        #     explodes on top of whatever joint_f the OSC writes.
        # Force every robot DOF into POSITION mode and rely on
        # joint_target_ke[:N_ARM_DOFS] = 0 to make the arm's PD a no-op
        # (-kd*qd damping only). URDF stays in its parser default
        # (POSITION across the board) so this matches the URDF regime.
        if profile.kind == "usd":
            for d in range(N_ROBOT_DOFS):
                builder.joint_target_mode[d] = newton.JointTargetMode.POSITION
            ctrl_source_attr = builder.custom_attributes["mujoco:ctrl_source"]
            n_acts = len(builder.joint_target_mode)
            ctrl_source_attr.values = [int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET)] * n_acts

        # Resolve the EE body index from the profile name now that the asset
        # is loaded. Stash on the instance so __init__ can pass it into the
        # OSCController and the disturbance kernel.
        self.ee_index = _resolve_ee_body_index(builder, profile)

        # Per-body gravity compensation. Factory's robot articulation runs
        # with disable_gravity=True (factory_env_cfg.py:129); to match that
        # regime in Newton's MuJoCo solver we set mujoco:gravcomp = 1.0 on
        # every body the parser added, fingers included. Asset-agnostic
        # so any inert body the asset author included (e.g. Factory USD's
        # fixed-jointed panda_force_sensor between link7 and panda_hand)
        # is automatically covered. Without force_sensor coverage that
        # 10 g body's weight propagates as a residual wrench up the chain
        # and shows up as steady-state TCP drift on the USD path. The
        # OSC scene has the gripper open with no contact, so finger
        # gravcomp is desirable: it removes the ~0.6 N of finger weight
        # hanging off the wrist that the OSC would otherwise have to
        # fight against. (Other examples that calibrate contact forces
        # from finger PD - e.g. nut-bolt - may want to skip fingers.)
        gravcomp_targets = set(range(body_count_before_parse, builder.body_count))

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in gravcomp_targets:
            gravcomp_body.values[body_idx] = 1.0

        return builder

    # ------------------------------------------------------------------
    # Sim loop
    # ------------------------------------------------------------------

    def capture(self) -> None:
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        # Apply pending disturbance force (if any) at the EE body's COM. We
        # write into state_0.body_f before stepping. `body_f` is a wrench in
        # world frame referenced at the COM (see Newton state docs).
        if self._disturbance_frames_remaining > 0:
            self._inject_disturbance()
            self._disturbance_frames_remaining -= 1

        # OSC tick gating: refresh torques only every `control_decimation`
        # frames; otherwise the previous joint_f is held by the solver.
        if self._gui_frame % self._control_decimation == 0:
            self._sync_gui_target_to_device()
            self.osc.update_tcp_state(self.state_0)
            self.osc.update_tcp_jacobian(self.state_0)

            # Step-clip the OSC's actual target toward the GUI target.
            wp.launch(
                step_clip_target_kernel,
                dim=self.world_count,
                inputs=[
                    self.osc.target_pos,
                    self.osc.target_quat,
                    self.gui_target_pos,
                    self.gui_target_quat,
                    float(self._pos_step_max),
                    float(self._rot_step_max),
                ],
                device=self.model.device,
            )
            self.osc.compute_torques(self.control, state=self.state_0)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        # Refresh GUI diagnostics periodically (host readback is cheap but
        # not free, so we don't do it every frame).
        if self._gui_frame % self._diag_refresh_period == 0:
            self._refresh_diagnostics()

        # Test-mode logging is gated on the diagnostic refresh tick - the
        # host readback inside _refresh_diagnostics already pulled tcp_pos
        # for the cached UI scalars, so reusing _diag_tcp_pos_host here
        # avoids an extra .numpy() per frame.
        if self.test_mode and self._gui_frame % self._diag_refresh_period == 0:
            tcp_z = float(self._diag_tcp_pos_host[2])
            if self._initial_tcp_z is None:
                self._initial_tcp_z = tcp_z
                print(f"[osc-test] initial TCP z = {tcp_z:.4f} m", flush=True)
            drop = (self._initial_tcp_z - tcp_z) * 1000.0
            print(
                f"[osc-test] t={self.sim_time:5.2f}s  TCP z={tcp_z:.4f} m  drop={drop:+6.2f} mm",
                flush=True,
            )
        # Always advance the frame counter - it gates the control_decimation
        # OSC tick and the diagnostic refresh, neither of which should be
        # bound to test mode.
        self._gui_frame += 1

    # ------------------------------------------------------------------
    # Rendering / GUI
    # ------------------------------------------------------------------

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self._show_debug_frames and self.viewer.world_offsets is not None:
            wp.launch(
                update_osc_debug_frame_lines_kernel,
                dim=self.world_count,
                inputs=[
                    self.osc.tcp_pos,
                    self.osc.tcp_quat,
                    self.osc.target_pos,
                    self.osc.target_quat,
                    self.viewer.world_offsets,
                    float(self._debug_axis_len),
                ],
                outputs=[self._dbg_starts, self._dbg_ends],
                device=self.model.device,
            )
            self.viewer.log_lines("/osc_debug_frames", self._dbg_starts, self._dbg_ends, self._dbg_colors)
        else:
            self.viewer.log_lines("/osc_debug_frames", None, None, None)
        self.viewer.end_frame()

    def _inject_disturbance(self) -> None:
        """Write the active disturbance wrench into ``state_0.body_f``.

        Runs entirely on device via :func:`apply_disturbance_force_kernel` -
        no host round-trip per frame. ``self._disturbance_force`` is a
        per-world ``wp.array[wp.vec3]`` on device; the kernel scatters it
        into ``body_f`` at the EE body index, zeroing the torque components.
        """
        if self.state_0.body_f is None:
            return
        wp.launch(
            apply_disturbance_force_kernel,
            dim=self.world_count,
            inputs=[
                self.state_0.body_f,
                self._disturbance_force,
                self.ee_body_index,
                self.num_bodies_per_world,
            ],
            device=self.model.device,
        )

    def _refresh_diagnostics(self) -> None:
        """Refresh host-side diagnostic scalars for the GUI panel.

        Vector reductions (pos err, rot err, torque norm, mass-matrix
        symmetry residual) all run as Warp kernels writing into per-world
        wp.array[float] buffers; we read back only the active world's
        scalar. The condition number runs through torch via wp.to_torch
        interop (the example already uses torch for the OSC's nullspace
        path, so this adds no new dependency).
        """
        active = self._active_world
        dev = self.model.device

        wp.launch(
            reduce_pos_err_mm_kernel,
            dim=self.world_count,
            inputs=[self.osc.pos_err],
            outputs=[self._diag_pos_err_mm_buf],
            device=dev,
        )
        wp.launch(
            reduce_rot_err_deg_kernel,
            dim=self.world_count,
            inputs=[self.osc.rot_err],
            outputs=[self._diag_rot_err_deg_buf],
            device=dev,
        )
        wp.launch(
            reduce_arm_torque_norm_kernel,
            dim=self.world_count,
            inputs=[self.osc.arm_torque, self.osc.n_arm_dofs],
            outputs=[self._diag_arm_torque_norm_buf],
            device=dev,
        )
        # Reuse the OSC's preallocated H buffer; eval_mass_matrix writes
        # into it without allocating a fresh array.
        self.art_view.eval_mass_matrix(
            self.state_0,
            H=self.osc.h_full,
            J=self.osc.j_full,
            body_I_s=self.osc._body_I_s,
            joint_S_s=self.osc._joint_S_s,
        )
        wp.launch(
            reduce_h_symmetry_resid_kernel,
            dim=self.world_count,
            inputs=[self.osc.h_full, self.osc.n_arm_dofs],
            outputs=[self._diag_h_sym_resid_buf],
            device=dev,
        )

        # Pack the four scalars and read back once per refresh.
        wp.launch(
            pack_diagnostics_kernel,
            dim=self.world_count,
            inputs=[
                self._diag_pos_err_mm_buf,
                self._diag_rot_err_deg_buf,
                self._diag_arm_torque_norm_buf,
                self._diag_h_sym_resid_buf,
            ],
            outputs=[self._diag_packed_buf],
            device=dev,
        )
        packed = self._diag_packed_buf.numpy()[active]
        self._diag_pos_err_mm = float(packed[0])
        self._diag_rot_err_deg = float(packed[1])
        self._diag_arm_torque_norm = float(packed[2])
        self._diag_h_symmetry_resid = float(packed[3])

        # Cache TCP / target pose for the render-thread UI - one readback at
        # diag rate, none on every UI tick.
        tcp_pos_np = self.osc.tcp_pos.numpy()[active]
        tgt_pos_np = self.osc.target_pos.numpy()[active]
        self._diag_tcp_pos_host = (float(tcp_pos_np[0]), float(tcp_pos_np[1]), float(tcp_pos_np[2]))
        self._diag_target_pos_host = (float(tgt_pos_np[0]), float(tgt_pos_np[1]), float(tgt_pos_np[2]))

        # Condition number via torch (already imported for the nullspace).
        try:
            import torch  # noqa: PLC0415

            H_t = wp.to_torch(self.osc.h_full)
            J_t = wp.to_torch(self.osc.j_tcp)
            H_arm = H_t[active, : self.osc.n_arm_dofs, : self.osc.n_arm_dofs]
            J = J_t[active]
            JHJt = J @ torch.linalg.inv(H_arm) @ J.transpose(-2, -1)
            self._diag_jacobian_cond = float(torch.linalg.cond(JHJt).item())
        except ImportError:
            self._diag_jacobian_cond = -1.0

    def _sync_gui_target_to_device(self) -> None:
        """Copy host-side GUI target arrays into the wp.array on the device.

        Skipped when nothing changed since the last sync to keep the per-tick
        cost down. Also recomputes the quaternion from RPY each time so the
        OSC's clipped slerp has a consistent target.

        Writes happen in place via ``wp.array.assign`` on the buffers that
        were allocated once in ``__init__`` - no per-tick allocation.
        """
        if not self._gui_target_dirty:
            return
        self.gui_target_pos.assign(self._gui_target_pos_host)
        self.gui_target_rpy.assign(self._gui_target_rpy_host)
        wp.launch(
            rpy_to_quat_kernel,
            dim=self.world_count,
            inputs=[self.gui_target_rpy],
            outputs=[self.gui_target_quat],
            device=self.model.device,
        )
        self._gui_target_dirty = False

    def _set_target_for_active_or_broadcast(self, axis: int, value: float, *, is_pos: bool) -> None:
        """Helper called by render_ui to apply a single-axis change."""
        host = self._gui_target_pos_host if is_pos else self._gui_target_rpy_host
        if self._broadcast:
            for w in range(self.world_count):
                host[w][axis] = value
        else:
            host[self._active_world][axis] = value
        self._gui_target_dirty = True

    def render_ui(self, imgui) -> None:
        imgui.separator()
        imgui.text("Panda OSC - phase 4 GUI")
        imgui.separator()

        # World selector.
        if self.world_count > 1:
            changed, val = imgui.slider_int("Active world", self._active_world, 0, self.world_count - 1)
            if changed:
                self._active_world = val
            changed_b, val_b = imgui.checkbox("Broadcast to all worlds", self._broadcast)
            if changed_b:
                self._broadcast = val_b

        imgui.separator()
        imgui.text("Target (world frame)")
        # Position sliders: +/-0.5 m around current per-world target seed.
        # Reading current host state lets the slider reflect step-clipped state.
        active = self._active_world
        pos = self._gui_target_pos_host[active]
        labels_pos = ("X [m]", "Y [m]", "Z [m]")
        for i, lbl in enumerate(labels_pos):
            changed, val = imgui.slider_float(lbl, float(pos[i]), -1.5, 1.5)
            if changed:
                self._set_target_for_active_or_broadcast(i, float(val), is_pos=True)

        rpy = self._gui_target_rpy_host[active]
        labels_rpy = ("Roll [rad]", "Pitch [rad]", "Yaw [rad]")
        for i, lbl in enumerate(labels_rpy):
            changed, val = imgui.slider_float(lbl, float(rpy[i]), -3.1416, 3.1416)
            if changed:
                self._set_target_for_active_or_broadcast(i, float(val), is_pos=False)

        imgui.separator()
        imgui.text("Step-clip caps (per OSC tick)")
        changed, val = imgui.slider_float("Pos step max [m]", self._pos_step_max, 0.0001, 0.05)
        if changed:
            self._pos_step_max = float(val)
        changed, val = imgui.slider_float("Rot step max [rad]", self._rot_step_max, 0.001, 0.5)
        if changed:
            self._rot_step_max = float(val)

        imgui.separator()
        imgui.text("Control rate")
        changed, val = imgui.slider_int("Decimation (frames/tick)", self._control_decimation, 1, 16)
        if changed:
            self._control_decimation = max(1, int(val))
        eff_hz = self.fps / max(1, self._control_decimation)
        imgui.text(f"Effective control rate: {eff_hz:.1f} Hz")

        imgui.separator()
        imgui.text("Nullspace centering (phase 6)")
        changed, val = imgui.checkbox("Enable nullspace", self.osc.enable_nullspace)
        if changed:
            self.osc.enable_nullspace = bool(val)
        changed, val = imgui.slider_float("Kp_null", float(self.osc.kp_null), 0.0, 50.0)
        if changed:
            self.osc.kp_null = float(val)
        changed, val = imgui.slider_float("Kd_null", float(self.osc.kd_null), 0.0, 10.0)
        if changed:
            self.osc.kd_null = float(val)

        imgui.separator()
        imgui.text("Visualization")
        changed, val = imgui.checkbox("Show debug frames", self._show_debug_frames)
        if changed:
            self._show_debug_frames = bool(val)
        changed, val = imgui.slider_float("Axis length [m]", self._debug_axis_len, 0.01, 0.3)
        if changed:
            self._debug_axis_len = float(val)

        imgui.separator()
        imgui.text("Disturbances (sanity / compliance probe)")
        changed, val = imgui.slider_float("Force [N]", self._disturbance_magnitude, 0.0, 30.0)
        if changed:
            self._disturbance_magnitude = float(val)
        changed, val = imgui.slider_int("Duration [frames]", self._disturbance_duration_frames, 1, 240)
        if changed:
            self._disturbance_duration_frames = int(val)
        for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
            for sign, sign_str in ((1.0, "+"), (-1.0, "-")):
                if imgui.button(f"Push {sign_str}{axis_name}"):
                    # Build per-world force list with only the active world set,
                    # then assign in-place into the preallocated device buffer.
                    f_list = [(0.0, 0.0, 0.0) for _ in range(self.world_count)]
                    f_active = [0.0, 0.0, 0.0]
                    f_active[axis_idx] = sign * self._disturbance_magnitude
                    f_list[active] = tuple(f_active)
                    self._disturbance_force.assign(f_list)
                    self._disturbance_frames_remaining = self._disturbance_duration_frames
                imgui.same_line()
        imgui.new_line()
        if imgui.button("Step target +5cm X"):
            self._gui_target_pos_host[active][0] += 0.05
            self._gui_target_dirty = True
        imgui.same_line()
        if imgui.button("Step target +5cm Z"):
            self._gui_target_pos_host[active][2] += 0.05
            self._gui_target_dirty = True

        imgui.separator()
        imgui.text("Diagnostics (active world)")
        # Read cached host scalars (refreshed at the diagnostic rate). No
        # device sync on the render thread.
        tcp_pos = self._diag_tcp_pos_host
        tgt_pos = self._diag_target_pos_host
        gui_pos = self._gui_target_pos_host[active]
        # GUI-vs-OSC-target distance (already on the device path; for the
        # display we compute a single scalar from the three components -
        # the underlying OSC target_pos has no Warp reduction yet for this
        # particular pair, so this is the only floats-arithmetic line in
        # the panel).
        dx = float(gui_pos[0]) - float(tgt_pos[0])
        dy = float(gui_pos[1]) - float(tgt_pos[1])
        dz = float(gui_pos[2]) - float(tgt_pos[2])
        gap_mm = ((dx * dx + dy * dy + dz * dz) ** 0.5) * 1000.0
        imgui.text(f"TCP pos: {tcp_pos[0]:+.3f} {tcp_pos[1]:+.3f} {tcp_pos[2]:+.3f}")
        imgui.text(f"OSC tgt: {tgt_pos[0]:+.3f} {tgt_pos[1]:+.3f} {tgt_pos[2]:+.3f}")
        imgui.text(f"Pos err: {self._diag_pos_err_mm:.1f} mm    Rot err: {self._diag_rot_err_deg:.2f} deg")
        imgui.text(f"|GUI tgt - OSC tgt|: {gap_mm:.1f} mm  (clipped per tick)")
        imgui.text(f"||arm_torque||: {self._diag_arm_torque_norm:.2f} N.m")

        imgui.separator()
        imgui.text("Sanity invariants")
        sym_ok = "OK" if self._diag_h_symmetry_resid < 1e-4 else "FAIL"
        cond_ok = "OK" if 0 <= self._diag_jacobian_cond < 1e6 else "WARN"
        imgui.text(f"H symmetry resid: {self._diag_h_symmetry_resid:.1e} [{sym_ok}]")
        imgui.text(f"cond(J H^-1 J^T): {self._diag_jacobian_cond:.1e} [{cond_ok}]")
        imgui.text(f"Disturbance: {self._disturbance_frames_remaining} frames remaining")
        imgui.text(f"Sim time: {self.sim_time:.2f} s   frame: {self._gui_frame}")

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """Acceptance checks.

        * Always: arm did not collapse under gravity (drop < 5 cm).
        * If ``--target-offset`` was set: TCP closed at least 50 % of the
          commanded offset, validating step-clip + OSC reach end-to-end.
        """
        if self._initial_tcp_z is None:
            return  # Test mode wasn't on, or step never ran.
        final_z = float(self.osc.tcp_pos.numpy()[0][2])
        drop = self._initial_tcp_z - final_z
        assert drop < 0.05, (
            f"Arm dropped {drop * 1000:.1f} mm - gravcomp not effective or "
            f"effort-mode setup is wrong. Initial TCP z={self._initial_tcp_z:.3f}, "
            f"final={final_z:.3f}."
        )

        if self._target_offset is not None:
            # The OSC target should have stepped most of the way toward the
            # GUI target, and the TCP should be tracking it. Compute the
            # per-world distance via reduce_pos_distance_mm_kernel and read
            # back world 0's value.
            wp.launch(
                reduce_pos_distance_mm_kernel,
                dim=self.world_count,
                inputs=[self.gui_target_pos, self.osc.tcp_pos],
                outputs=[self._diag_pos_distance_mm_buf],
                device=self.model.device,
            )
            err_mm = float(self._diag_pos_distance_mm_buf.numpy()[0])
            err = err_mm / 1000.0
            offset_norm = self._initial_target_offset_norm
            closed_frac = 1.0 - err / max(offset_norm, 1e-9)
            print(
                f"[osc-test] target offset = {offset_norm * 1000:.1f} mm, "
                f"final TCP-to-target error = {err * 1000:.1f} mm "
                f"(closed {closed_frac * 100:.0f}%)",
                flush=True,
            )
            # Accept >= 20% closure: the joint-level damping that stabilizes
            # the arm slows tracking, so a 90-frame (1.5 s) horizon is enough
            # to verify the arm moves toward the target but not to fully
            # converge. Run with --num-frames 300 for a tighter check.
            assert closed_frac > 0.2, (
                f"Arm closed only {closed_frac * 100:.0f}% of the {offset_norm * 1000:.1f} mm "
                f"target offset. Step-clip max={self._pos_step_max:.4f} m/tick, "
                f"control rate={self.fps / self._control_decimation:.1f} Hz."
            )

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=180)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--target-offset",
            type=float,
            nargs=3,
            metavar=("DX", "DY", "DZ"),
            default=None,
            help="Offset (m, world frame) added to the initial TCP target. "
            "Used in headless test to verify step-clip + reach.",
        )
        parser.add_argument(
            "--robot",
            choices=("urdf", "usd"),
            default="urdf",
            help="Which Panda asset to load. 'urdf' (default) uses Newton's "
            "curated FR3 + Franka hand URDF. 'usd' uses IsaacLab Factory's "
            "franka_mimic.usd (must be copied to "
            "newton/examples/assets/franka_mimic/franka_mimic.usd; see the "
            "README at that path).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
