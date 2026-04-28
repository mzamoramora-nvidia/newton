# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Operational-Space Controller (OSC) for the Franka Panda example.

Mirrors the torque-level OSC used in IsaacLab's Factory tasks (see
``isaaclab_tasks/direct/factory/factory_control.py``). The controller reads
mass matrix and Jacobian from Newton's :class:`~newton.selection.ArticulationView`
and emits joint torques into ``control.joint_f``.

Frame-shift helpers handle Newton's COM-velocity convention: ``body_qd`` and the
spatial Jacobian linear rows are referenced to each link's COM. Operational-space
control needs them at the TCP. Use :func:`shift_jacobian_com_to_tcp` and
:func:`shift_velocity_com_to_tcp`.

Phase-1 stub: ``OSCController.compute_torques`` is implemented in phase 2; for
now it writes zeros so gravity compensation alone holds the arm.
"""

from __future__ import annotations

import warp as wp


# ---------------------------------------------------------------------------
# Frame-shift helpers
#
# Newton convention recap (newton/_src/sim/state.py, model.py):
#   body_q[i]   : pose of the **link** frame (URDF body origin), world frame.
#   body_com[i] : COM offset in **local link** frame.
#                 r_com_world = transform_point(body_q[i], body_com[i]).
#   body_qd[i]  : (v_com_world, omega_world). Linear part is COM velocity.
#   eval_jacobian: J such that J @ qd == body_qd. So linear rows are in the
#                  COM-velocity convention too.
#
# To go from COM-velocity (Newton) to TCP-velocity (what OSC needs):
#   v_tcp = v_com + omega x (r_tcp - r_com)
#   J_tcp_lin = J_com_lin - skew(r_tcp - r_com) @ J_ang
# Angular components are unchanged.
# ---------------------------------------------------------------------------


@wp.func
def _skew(v: wp.vec3) -> wp.mat33:
    return wp.mat33(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )


@wp.kernel(enable_backward=False)
def compute_tcp_pose_and_velocity_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com_local: wp.array[wp.vec3],
    ee_body_index: int,
    num_bodies_per_world: int,
    tcp_offset_local: wp.transform,
    tcp_pos: wp.array[wp.vec3],
    tcp_quat: wp.array[wp.vec4],
    tcp_linvel: wp.array[wp.vec3],
    tcp_angvel: wp.array[wp.vec3],
    com_world: wp.array[wp.vec3],
):
    """Per-world: extract TCP pose + velocity from body_q / body_qd.

    Outputs:
        tcp_pos[w]     : TCP world position [m]
        tcp_quat[w]    : TCP world orientation, xyzw
        tcp_linvel[w]  : TCP world linear velocity [m/s] (shifted from COM)
        tcp_angvel[w]  : world angular velocity [rad/s] (link == body)
        com_world[w]   : EE link COM in world (cached for Jacobian shift)
    """
    w = wp.tid()
    body_idx = w * num_bodies_per_world + ee_body_index
    link_xform = body_q[body_idx]

    # TCP world pose = link pose composed with local TCP offset.
    tcp_xform = wp.transform_multiply(link_xform, tcp_offset_local)
    p_tcp = wp.transform_get_translation(tcp_xform)
    q_tcp = wp.transform_get_rotation(tcp_xform)
    tcp_pos[w] = p_tcp
    tcp_quat[w] = wp.vec4(q_tcp[0], q_tcp[1], q_tcp[2], q_tcp[3])

    # COM world position: r_com_w = transform_point(link_xform, body_com_local[ee])
    r_com_w = wp.transform_point(link_xform, body_com_local[body_idx])
    com_world[w] = r_com_w

    # Velocity shift: v_tcp = v_com + omega x (r_tcp - r_com)
    twist = body_qd[body_idx]
    v_com = wp.vec3(twist[0], twist[1], twist[2])
    omega = wp.vec3(twist[3], twist[4], twist[5])
    tcp_linvel[w] = v_com + wp.cross(omega, p_tcp - r_com_w)
    tcp_angvel[w] = omega


@wp.kernel(enable_backward=False)
def update_osc_debug_frame_lines_kernel(
    tcp_pos: wp.array[wp.vec3],
    tcp_quat: wp.array[wp.vec4],
    target_pos: wp.array[wp.vec3],
    target_quat: wp.array[wp.vec4],
    world_offsets: wp.array[wp.vec3],
    axis_len: float,
    # output: 7 * world_count segments per world (3 axes for TCP + 3 for target + 1 error line).
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    """Build line segments for the OSC debug frame overlay.

    Per world, 7 segments in this order:
      0..2: TCP current frame X/Y/Z axes (rendered RGB).
      3..5: TCP target frame X/Y/Z axes (rendered half-saturation RGB).
      6:    error vector from TCP to target.
    """
    w = wp.tid()
    off = world_offsets[w]
    base = w * 7

    p_tcp = tcp_pos[w] + off
    q_tcp = wp.quat(tcp_quat[w][0], tcp_quat[w][1], tcp_quat[w][2], tcp_quat[w][3])
    starts[base + 0] = p_tcp
    ends[base + 0] = p_tcp + wp.quat_rotate(q_tcp, wp.vec3(axis_len, 0.0, 0.0))
    starts[base + 1] = p_tcp
    ends[base + 1] = p_tcp + wp.quat_rotate(q_tcp, wp.vec3(0.0, axis_len, 0.0))
    starts[base + 2] = p_tcp
    ends[base + 2] = p_tcp + wp.quat_rotate(q_tcp, wp.vec3(0.0, 0.0, axis_len))

    p_tgt = target_pos[w] + off
    q_tgt = wp.quat(target_quat[w][0], target_quat[w][1], target_quat[w][2], target_quat[w][3])
    starts[base + 3] = p_tgt
    ends[base + 3] = p_tgt + wp.quat_rotate(q_tgt, wp.vec3(axis_len, 0.0, 0.0))
    starts[base + 4] = p_tgt
    ends[base + 4] = p_tgt + wp.quat_rotate(q_tgt, wp.vec3(0.0, axis_len, 0.0))
    starts[base + 5] = p_tgt
    ends[base + 5] = p_tgt + wp.quat_rotate(q_tgt, wp.vec3(0.0, 0.0, axis_len))

    starts[base + 6] = p_tcp
    ends[base + 6] = p_tgt


@wp.kernel(enable_backward=False)
def rpy_to_quat_kernel(
    rpy: wp.array[wp.vec3],
    quat_out: wp.array[wp.vec4],
):
    """Convert per-world (roll, pitch, yaw) [rad] to xyzw quaternion."""
    w = wp.tid()
    q = wp.quat_rpy(rpy[w][0], rpy[w][1], rpy[w][2])
    quat_out[w] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel(enable_backward=False)
def quat_to_rpy_kernel(
    quat: wp.array[wp.vec4],
    rpy_out: wp.array[wp.vec3],
):
    """Convert per-world xyzw quaternion to (roll, pitch, yaw) [rad]."""
    w = wp.tid()
    q = wp.quat(quat[w][0], quat[w][1], quat[w][2], quat[w][3])
    rpy_out[w] = wp.quat_to_rpy(q)


@wp.kernel(enable_backward=False)
def step_clip_target_kernel(
    actual_pos: wp.array[wp.vec3],
    actual_quat: wp.array[wp.vec4],
    gui_pos: wp.array[wp.vec3],
    gui_quat: wp.array[wp.vec4],
    max_pos_step: float,
    max_rot_step: float,
):
    """Move actual target toward GUI target by at most one step per call.

    Mirrors Factory's per-tick action clamping: the OSC sees a smooth target
    even when the GUI slider jumps. Position uses a linear interpolation
    capped at ``max_pos_step`` per tick; orientation uses slerp capped at
    ``max_rot_step``.
    """
    w = wp.tid()

    # Position step.
    delta = gui_pos[w] - actual_pos[w]
    dist = wp.length(delta)
    if dist > max_pos_step:
        delta = delta * (max_pos_step / dist)
    actual_pos[w] = actual_pos[w] + delta

    # Orientation step (slerp with per-tick cap).
    qa = wp.quat(actual_quat[w][0], actual_quat[w][1], actual_quat[w][2], actual_quat[w][3])
    qg = wp.quat(gui_quat[w][0], gui_quat[w][1], gui_quat[w][2], gui_quat[w][3])
    dot_q = qa[0] * qg[0] + qa[1] * qg[1] + qa[2] * qg[2] + qa[3] * qg[3]
    if dot_q < 0.0:
        qg = wp.quat(-qg[0], -qg[1], -qg[2], -qg[3])
        dot_q = -dot_q
    angle = 2.0 * wp.acos(wp.clamp(dot_q, -1.0, 1.0))
    if angle < 1e-6:
        return
    t = float(1.0)
    if angle > max_rot_step:
        t = max_rot_step / angle
    q_new = wp.quat_slerp(qa, qg, t)
    actual_quat[w] = wp.vec4(q_new[0], q_new[1], q_new[2], q_new[3])


@wp.kernel(enable_backward=False)
def reduce_pos_err_mm_kernel(
    pos_err: wp.array[wp.vec3],
    out_mm: wp.array[float],
):
    """Per-world: ||pos_err|| in millimetres."""
    w = wp.tid()
    out_mm[w] = wp.length(pos_err[w]) * 1000.0


@wp.kernel(enable_backward=False)
def reduce_rot_err_deg_kernel(
    rot_err: wp.array[wp.vec3],
    out_deg: wp.array[float],
):
    """Per-world: ||rot_err|| (radians) converted to degrees via wp.degrees."""
    w = wp.tid()
    out_deg[w] = wp.degrees(wp.length(rot_err[w]))


@wp.kernel(enable_backward=False)
def reduce_arm_torque_norm_kernel(
    arm_torque: wp.array2d(dtype=float),  # (W, n_arm_dofs)
    n_arm_dofs: int,
    out_norm: wp.array[float],
):
    """Per-world: Euclidean norm of the arm-DOF torque vector."""
    w = wp.tid()
    s = float(0.0)
    for d in range(n_arm_dofs):
        v = arm_torque[w, d]
        s += v * v
    out_norm[w] = wp.sqrt(s)


@wp.kernel(enable_backward=False)
def reduce_h_symmetry_resid_kernel(
    h_full: wp.array3d(dtype=float),  # (W, n_dofs, n_dofs)
    n_arm_dofs: int,
    out: wp.array[float],
):
    """Frobenius residual ``||H - H^T||_F / ||H||_F`` over the arm sub-block."""
    w = wp.tid()
    sym_sq = float(0.0)
    norm_sq = float(0.0)
    for i in range(n_arm_dofs):
        for j in range(n_arm_dofs):
            a = h_full[w, i, j]
            b = h_full[w, j, i]
            d = a - b
            sym_sq += d * d
            norm_sq += a * a
    out[w] = wp.sqrt(sym_sq) / wp.max(wp.sqrt(norm_sq), 1.0e-9)


@wp.kernel(enable_backward=False)
def reduce_pos_distance_mm_kernel(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    out_mm: wp.array[float],
):
    """Per-world: ``||a - b||`` in millimetres."""
    w = wp.tid()
    out_mm[w] = wp.length(a[w] - b[w]) * 1000.0


@wp.kernel(enable_backward=False)
def apply_disturbance_force_kernel(
    body_f: wp.array[wp.spatial_vector],
    disturbance_force: wp.array[wp.vec3],
    ee_body_index: int,
    num_bodies_per_world: int,
):
    """Write ``disturbance_force[w]`` into ``body_f`` at the EE body's COM.

    ``body_f`` is a wrench in world frame referenced at COM (see Newton state
    docs). Torque components are zeroed.
    """
    w = wp.tid()
    body_idx = w * num_bodies_per_world + ee_body_index
    f = disturbance_force[w]
    body_f[body_idx] = wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def compute_pose_error_kernel(
    tcp_pos: wp.array[wp.vec3],
    tcp_quat: wp.array[wp.vec4],
    target_pos: wp.array[wp.vec3],
    target_quat: wp.array[wp.vec4],
    pos_err: wp.array[wp.vec3],
    rot_err: wp.array[wp.vec3],
):
    """Pose error in TCP world frame.

    pos_err = target - current.
    rot_err = axis-angle of (target * current^-1), short-path corrected.
    """
    w = wp.tid()
    pos_err[w] = target_pos[w] - tcp_pos[w]

    qc = wp.quat(tcp_quat[w][0], tcp_quat[w][1], tcp_quat[w][2], tcp_quat[w][3])
    qt = wp.quat(target_quat[w][0], target_quat[w][1], target_quat[w][2], target_quat[w][3])

    # Short path: flip qt if dot(qc, qt) < 0.
    dot = qc[0] * qt[0] + qc[1] * qt[1] + qc[2] * qt[2] + qc[3] * qt[3]
    if dot < 0.0:
        qt = wp.quat(-qt[0], -qt[1], -qt[2], -qt[3])

    q_err = wp.mul(qt, wp.quat_inverse(qc))
    axis, angle = wp.quat_to_axis_angle(q_err)
    rot_err[w] = axis * angle


@wp.kernel(enable_backward=False)
def apply_task_space_pd_kernel(
    pos_err: wp.array[wp.vec3],
    rot_err: wp.array[wp.vec3],
    tcp_linvel: wp.array[wp.vec3],
    tcp_angvel: wp.array[wp.vec3],
    kp: wp.array2d(dtype=float),  # (world_count, 6)
    kd: wp.array2d(dtype=float),  # (world_count, 6)
    wrench: wp.array2d(dtype=float),  # (world_count, 6)
):
    """F = Kp * e - Kd * v, with desired velocity = 0."""
    w = wp.tid()
    # Linear (translational) components.
    wrench[w, 0] = kp[w, 0] * pos_err[w][0] - kd[w, 0] * tcp_linvel[w][0]
    wrench[w, 1] = kp[w, 1] * pos_err[w][1] - kd[w, 1] * tcp_linvel[w][1]
    wrench[w, 2] = kp[w, 2] * pos_err[w][2] - kd[w, 2] * tcp_linvel[w][2]
    # Angular (rotational) components.
    wrench[w, 3] = kp[w, 3] * rot_err[w][0] - kd[w, 3] * tcp_angvel[w][0]
    wrench[w, 4] = kp[w, 4] * rot_err[w][1] - kd[w, 4] * tcp_angvel[w][1]
    wrench[w, 5] = kp[w, 5] * rot_err[w][2] - kd[w, 5] * tcp_angvel[w][2]


@wp.kernel(enable_backward=False)
def map_wrench_to_arm_torque_kernel(
    j_tcp: wp.array3d(dtype=float),  # (world_count, 6, n_arm_dofs)
    wrench: wp.array2d(dtype=float),  # (world_count, 6)
    n_arm_dofs: int,
    arm_torque: wp.array2d(dtype=float),  # (world_count, n_arm_dofs)
):
    """tau = J_tcp^T @ F (per world)."""
    w, d = wp.tid()
    if d >= n_arm_dofs:
        return
    s = float(0.0)
    for k in range(6):
        s += j_tcp[w, k, d] * wrench[w, k]
    arm_torque[w, d] = s


@wp.kernel(enable_backward=False)
def scatter_arm_torque_to_joint_f_kernel(
    arm_torque: wp.array2d(dtype=float),  # (world_count, n_arm_dofs)
    n_arm_dofs: int,
    n_dofs_per_world: int,
    joint_f: wp.array[float],  # flat (world_count * n_dofs_per_world,) - actually total model DOFs
):
    """Write the per-world arm torque into the flat joint_f array.

    joint_f layout matches joint_q: per-world DOFs laid out in sequence. Within
    each world, arm DOFs occupy indices 0 .. n_arm_dofs-1 of that world's slice.
    Other DOFs (gripper, free objects) are left untouched.
    """
    w, d = wp.tid()
    if d >= n_arm_dofs:
        return
    flat_idx = w * n_dofs_per_world + d
    joint_f[flat_idx] = arm_torque[w, d]


@wp.kernel(enable_backward=False)
def shift_jacobian_com_to_tcp_kernel(
    j_full: wp.array3d(dtype=float),  # (world_count, n_joints_per_art*6, n_dofs)
    com_world: wp.array[wp.vec3],
    tcp_pos: wp.array[wp.vec3],
    ee_joint_in_art: int,
    n_arm_dofs: int,
    j_tcp: wp.array3d(dtype=float),  # (world_count, 6, n_arm_dofs)
):
    """Slice EE rows from the per-joint Jacobian and shift linear part to TCP.

    Newton's spatial Jacobian is indexed by **joint position within the
    articulation** - not by global body index. Each joint contributes 6 rows
    (linear + angular) describing the spatial velocity at the child body's
    COM. Linear rows are world-frame velocity referenced at the COM; angular
    rows are world-frame angular velocity.

    j_tcp output: 6 x n_arm_dofs slice for the EE joint's child, with linear
    rows shifted from COM to TCP via J_lin_tcp = J_lin_com - skew(r_tcp - r_com) @ J_ang.
    """
    w, dof = wp.tid()
    if dof >= n_arm_dofs:
        return

    base = ee_joint_in_art * 6
    # Read angular rows first (unchanged).
    wx = j_full[w, base + 3, dof]
    wy = j_full[w, base + 4, dof]
    wz = j_full[w, base + 5, dof]
    j_tcp[w, 3, dof] = wx
    j_tcp[w, 4, dof] = wy
    j_tcp[w, 5, dof] = wz

    # Linear rows: shift by -skew(r_tcp - r_com) @ omega_col.
    delta = tcp_pos[w] - com_world[w]
    vx = j_full[w, base + 0, dof]
    vy = j_full[w, base + 1, dof]
    vz = j_full[w, base + 2, dof]

    # -skew(delta) @ (wx, wy, wz) = (-(dy*wz - dz*wy), -(dz*wx - dx*wz), -(dx*wy - dy*wx))
    #   = (dz*wy - dy*wz, dx*wz - dz*wx, dy*wx - dx*wy)
    dx = delta[0]
    dy = delta[1]
    dz = delta[2]
    j_tcp[w, 0, dof] = vx + (dz * wy - dy * wz)
    j_tcp[w, 1, dof] = vy + (dx * wz - dz * wx)
    j_tcp[w, 2, dof] = vz + (dy * wx - dx * wy)


# ---------------------------------------------------------------------------
# OSC controller
# ---------------------------------------------------------------------------


class OSCController:
    """Factory-style task-space PD with optional nullspace centering.

    Reads mass matrix and Jacobian from a :class:`~newton.selection.ArticulationView`
    each step, slices arm DOFs and shifts the Jacobian to the TCP frame, then
    computes :math:`\\tau = J^T (K_p e - K_d v) + N^T M u_{null}`.

    Phase-1 stub: :meth:`compute_torques` writes zeros. Real implementation
    lands in phase 2 alongside the finite-difference correctness tests.

    Args:
        model: Finalized Newton :class:`~newton.Model`.
        articulation_view: View matching the robot articulation.
        world_count: Number of replicated worlds.
        ee_body_index: Per-world local body index of the EE link.
        tcp_offset_local: Pose of the TCP frame relative to the EE link frame.
        n_arm_dofs: Number of arm DOFs (7 for Franka).
        num_bodies_per_world: Bodies per replicated world.
        n_dofs_per_world: Total DOFs per replicated world (arm + gripper, 9 for Franka).
    """

    def __init__(
        self,
        model,
        articulation_view,
        world_count: int,
        ee_body_index: int,
        tcp_offset_local: wp.transform,
        n_arm_dofs: int,
        num_bodies_per_world: int,
        n_dofs_per_world: int,
        device: str | None = None,
    ):
        self.model = model
        self.view = articulation_view
        self.world_count = world_count
        self.ee_body_index = ee_body_index
        self.tcp_offset_local = tcp_offset_local
        self.n_arm_dofs = n_arm_dofs
        self.num_bodies_per_world = num_bodies_per_world
        self.n_dofs_per_world = n_dofs_per_world
        self.device = device or model.device

        # Resolve the joint position-within-articulation that maps to the EE
        # body. Newton's Jacobian is indexed by joint-within-art, not global
        # body, so we need this lookup once at init.
        articulation_start = model.articulation_start.numpy()
        joint_child = model.joint_child.numpy()
        art_start = int(articulation_start[0])
        art_end = int(articulation_start[1])
        ee_joint_in_art = -1
        for j in range(art_start, art_end):
            if int(joint_child[j]) == ee_body_index:
                ee_joint_in_art = j - art_start
                break
        if ee_joint_in_art < 0:
            raise RuntimeError(
                f"OSCController: no joint in articulation 0 has child body {ee_body_index}. "
                f"Verify ee_body_index is the global per-world index of the EE link."
            )
        self.ee_joint_in_art = ee_joint_in_art

        # Per-world TCP state buffers.
        self.tcp_pos = wp.zeros(world_count, dtype=wp.vec3, device=self.device)
        self.tcp_quat = wp.zeros(world_count, dtype=wp.vec4, device=self.device)
        self.tcp_linvel = wp.zeros(world_count, dtype=wp.vec3, device=self.device)
        self.tcp_angvel = wp.zeros(world_count, dtype=wp.vec3, device=self.device)
        self.com_world = wp.zeros(world_count, dtype=wp.vec3, device=self.device)

        # TCP-frame Jacobian (6 x n_arm_dofs per world).
        self.j_tcp = wp.zeros((world_count, 6, n_arm_dofs), dtype=float, device=self.device)

        # Targets (initialized to current TCP at first update).
        self.target_pos = wp.zeros(world_count, dtype=wp.vec3, device=self.device)
        self.target_quat = wp.zeros(world_count, dtype=wp.vec4, device=self.device)

        # Gains: per-world (6,) - [Fx, Fy, Fz, Tx, Ty, Tz].
        self.kp = wp.zeros((world_count, 6), dtype=float, device=self.device)
        self.kd = wp.zeros((world_count, 6), dtype=float, device=self.device)

        # Per-step intermediates.
        self.pos_err = wp.zeros(world_count, dtype=wp.vec3, device=self.device)
        self.rot_err = wp.zeros(world_count, dtype=wp.vec3, device=self.device)
        self.wrench = wp.zeros((world_count, 6), dtype=float, device=self.device)
        self.arm_torque = wp.zeros((world_count, n_arm_dofs), dtype=float, device=self.device)

        # Nullspace state (phase 6). See `enable_nullspace` to toggle.
        self.enable_nullspace = False
        self.q_default = wp.zeros((world_count, n_arm_dofs), dtype=float, device=self.device)
        self.kp_null = 5.0  # scalar gain on (q_default - q)
        self.kd_null = 1.0  # scalar gain on -qd

    def update_tcp_state(self, state) -> None:
        """Refresh per-world TCP pose, velocity, and EE-link COM from ``state``."""
        wp.launch(
            compute_tcp_pose_and_velocity_kernel,
            dim=self.world_count,
            inputs=[
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.ee_body_index,
                self.num_bodies_per_world,
                self.tcp_offset_local,
            ],
            outputs=[
                self.tcp_pos,
                self.tcp_quat,
                self.tcp_linvel,
                self.tcp_angvel,
                self.com_world,
            ],
            device=self.device,
        )

    def update_tcp_jacobian(self, state) -> None:
        """Compute the full Jacobian via ArticulationView and shift to TCP frame."""
        # ArticulationView.eval_jacobian returns shape
        # (articulation_count == world_count, n_joints_per_art*6, n_dofs).
        j_full = self.view.eval_jacobian(state)
        wp.launch(
            shift_jacobian_com_to_tcp_kernel,
            dim=(self.world_count, self.n_arm_dofs),
            inputs=[
                j_full,
                self.com_world,
                self.tcp_pos,
                self.ee_joint_in_art,
                self.n_arm_dofs,
            ],
            outputs=[self.j_tcp],
            device=self.device,
        )

    def set_target(self, target_pos: wp.array, target_quat: wp.array) -> None:
        """Set per-world TCP target pose.

        Args:
            target_pos: ``wp.array[wp.vec3]`` of length ``world_count``, world-frame [m].
            target_quat: ``wp.array[wp.vec4]`` of length ``world_count``, xyzw.
        """
        wp.copy(self.target_pos, target_pos)
        wp.copy(self.target_quat, target_quat)

    def set_gains(self, kp: wp.array, kd: wp.array) -> None:
        """Set per-world task-space gains.

        Args:
            kp: ``(world_count, 6)`` proportional gains, [Fx,Fy,Fz, Tx,Ty,Tz] order.
            kd: ``(world_count, 6)`` derivative gains.
        """
        wp.copy(self.kp, kp)
        wp.copy(self.kd, kd)

    def set_default_pose(self, q_default: wp.array) -> None:
        """Set the per-world default arm configuration used by nullspace centering.

        Args:
            q_default: ``wp.array`` shape ``(world_count, n_arm_dofs)`` [rad].
        """
        wp.copy(self.q_default, q_default)

    def _add_nullspace_torque(self, state) -> None:
        """Add Khatib-style nullspace centering to ``arm_torque`` (PyTorch path).

        tau_null = N^T H u_null,
            where N^T = I - J^T Jbar,  Jbar = Lambda J H^-1,  Lambda = (J H^-1 J^T)^-1.

        Uses Warp/Torch interop because the per-step matrix work involves 6x6
        and 7x7 inversions that are easier to express in PyTorch than in Warp
        tile kernels. Both paths share GPU memory via ``wp.to_torch``.
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return  # nullspace silently disabled when torch is unavailable.

        H_full = self.view.eval_mass_matrix(state)  # (W, n_dofs, n_dofs)
        H_t = wp.to_torch(H_full)
        H_arm = H_t[..., : self.n_arm_dofs, : self.n_arm_dofs]
        J = wp.to_torch(self.j_tcp)  # (W, 6, 7)
        arm_torque_t = wp.to_torch(self.arm_torque)  # (W, 7), shared memory

        # Pull arm joint state from the provided state.
        q_full = wp.to_torch(state.joint_q)
        qd_full = wp.to_torch(state.joint_qd)
        n_dofs_per_world = q_full.shape[0] // self.world_count
        q_arm = q_full.view(self.world_count, n_dofs_per_world)[:, : self.n_arm_dofs]
        qd_arm = qd_full.view(self.world_count, n_dofs_per_world)[:, : self.n_arm_dofs]

        q_default = wp.to_torch(self.q_default)

        H_inv = torch.linalg.inv(H_arm)
        Jt = J.transpose(-2, -1)
        Lambda = torch.linalg.inv(J @ H_inv @ Jt)
        Jbar = Lambda @ J @ H_inv  # (W, 6, 7)
        eye = torch.eye(self.n_arm_dofs, device=H_arm.device, dtype=H_arm.dtype)
        N_T = eye.expand_as(H_arm) - Jt @ Jbar  # (W, 7, 7)

        u_null = self.kp_null * (q_default - q_arm) - self.kd_null * qd_arm  # (W, 7)
        tau_null = (N_T @ (H_arm @ u_null.unsqueeze(-1))).squeeze(-1)  # (W, 7)

        arm_torque_t.add_(tau_null)
        # torch and Warp use separate CUDA streams. Synchronize so the next
        # Warp kernel (scatter_arm_torque_to_joint_f_kernel) sees the updated
        # arm_torque rather than reading stale memory.
        torch.cuda.synchronize()

    def compute_torques(self, control, state=None) -> None:
        """Compute arm torques from current TCP state, target, and gains.

        Writes ``control.joint_f`` arm slice (DOFs 0..n_arm_dofs-1 within each world).
        Other DOFs (gripper, free bodies in scene) are left untouched. Call
        :meth:`update_tcp_state` and :meth:`update_tcp_jacobian` first.

        Args:
            control: ``Control`` whose ``joint_f`` will receive the arm torques.
            state: Required when ``enable_nullspace`` is True (used to fetch H
                and joint state for the nullspace term).
        """
        # 1. Pose error (per world).
        wp.launch(
            compute_pose_error_kernel,
            dim=self.world_count,
            inputs=[self.tcp_pos, self.tcp_quat, self.target_pos, self.target_quat],
            outputs=[self.pos_err, self.rot_err],
            device=self.device,
        )
        # 2. Task-space PD wrench.
        wp.launch(
            apply_task_space_pd_kernel,
            dim=self.world_count,
            inputs=[
                self.pos_err,
                self.rot_err,
                self.tcp_linvel,
                self.tcp_angvel,
                self.kp,
                self.kd,
            ],
            outputs=[self.wrench],
            device=self.device,
        )
        # 3. tau_arm = J_tcp^T @ wrench  (per world, n_arm_dofs entries).
        wp.launch(
            map_wrench_to_arm_torque_kernel,
            dim=(self.world_count, self.n_arm_dofs),
            inputs=[self.j_tcp, self.wrench, self.n_arm_dofs],
            outputs=[self.arm_torque],
            device=self.device,
        )
        # 3b. Optional Khatib nullspace centering - adds onto arm_torque.
        if self.enable_nullspace:
            if state is None:
                raise RuntimeError("compute_torques: state must be provided when enable_nullspace=True")
            self._add_nullspace_torque(state)
        # 4. Scatter into the flat joint_f array (skip non-robot DOFs).
        if control.joint_f is None:
            raise RuntimeError("control.joint_f is None - does the model have any DOFs?")
        wp.launch(
            scatter_arm_torque_to_joint_f_kernel,
            dim=(self.world_count, self.n_arm_dofs),
            inputs=[self.arm_torque, self.n_arm_dofs, self.n_dofs_per_world],
            outputs=[control.joint_f],
            device=self.device,
        )
