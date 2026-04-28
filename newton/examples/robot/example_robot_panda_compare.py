# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Panda Compare
#
# Side-by-side URDF-vs-USD viewer to diagnose frame-alignment differences
# between Newton's Panda URDF and IsaacLab Factory's panda asset.
#
# Robot A: Newton URDF (newton-assets fr3_franka_hand.urdf).
# Robot B: IsaacLab Factory's franka_mimic.usd. The USD file is not
#   bundled with this repo for licensing reasons - install it locally at
#   newton/examples/assets/franka_mimic/franka_mimic.usd (see the README
#   in that directory for instructions). The example exits with a clear
#   message if the USD is missing.
#
# A shared joint-slider set drives both robots in mirror mode so they
# run identical kinematics. Per-link world-position deltas (re-based to
# Robot A) tabulate the canonical Panda chain. RGB axes are drawn at
# each robot's TCP plus a configurable "Factory-equivalent" frame on
# both robots so the user can verify a tunable hand-local offset places
# the magenta frame at the same physical point on each asset.
#
# Command: python -m newton.examples robot_panda_compare
#
###########################################################################

from __future__ import annotations

import os

import warp as wp

import newton
import newton.examples
import newton.utils


# Robot B always loads IsaacLab Factory's franka_mimic.usd from the
# Newton assets folder. The USD itself is not committed - see
# newton/examples/assets/franka_mimic/README.md for how to install it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FACTORY_USD = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "assets", "franka_mimic", "franka_mimic.usd")
)


# Robot base positions (m, world frame). Spaced apart so the two arms
# don't visually overlap in the default zero pose.
ROBOT_A_BASE = wp.vec3(-0.4, 0.0, 0.0)
ROBOT_B_BASE = wp.vec3(0.4, 0.0, 0.0)

# Per-robot indexing (matches example_robot_panda_osc.py).
N_BODIES_PER_ROBOT = 14  # base + fr3_link0..7 + fr3_link8 + fr3_hand + fr3_hand_tcp + 2 fingers
N_ARM_DOFS = 7
N_FINGER_DOFS = 2
N_DOFS_PER_ROBOT = N_ARM_DOFS + N_FINGER_DOFS

# Body offsets within one robot's slice.
HAND_BODY_OFFSET = 10  # fr3_hand
EE_BODY_OFFSET = 11    # fr3_hand_tcp

# Initial joint configuration (radians) - lifted from the OSC example
# so we share a common starting pose for visual comparison.
INIT_ARM_Q = (
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
)
INIT_FINGER_Q = (0.05, 0.05)


@wp.kernel(enable_backward=False)
def update_compare_frame_lines_kernel(
    body_q: wp.array[wp.transform],
    a_tcp_body_idx: int,
    b_tcp_body_idx: int,
    a_hand_body_idx: int,
    b_hand_body_idx: int,
    factory_offset: wp.transform,
    world_axis_len: float,
    axis_len: float,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    """Build the 15 line segments rendered as the comparison overlay.

    Layout (in starts/ends arrays):
      0..2  : World-origin X/Y/Z axes                   (full RGB, longer axis_len)
      3..5  : Robot A fr3_hand_tcp X/Y/Z axes           (full RGB)
      6..8  : Robot B fr3_hand_tcp X/Y/Z axes           (half-saturation RGB)
      9..11 : Factory-equivalent TCP on Robot A         (magenta-ish RGB)
      12..14: Factory-equivalent TCP on Robot B         (magenta-ish RGB)

    Factory-equivalent frames = body_q[<robot>_hand] * factory_offset, drawn
    on both robots so the magenta overlap reveals whether the user-tunable
    offset puts the TCP at the same physical point on each asset.
    """
    tid = wp.tid()
    if tid != 0:
        return

    # World frame at origin (axes only - origin point is implicit).
    p0 = wp.vec3(0.0, 0.0, 0.0)
    starts[0] = p0
    ends[0] = wp.vec3(world_axis_len, 0.0, 0.0)
    starts[1] = p0
    ends[1] = wp.vec3(0.0, world_axis_len, 0.0)
    starts[2] = p0
    ends[2] = wp.vec3(0.0, 0.0, world_axis_len)

    # Robot A TCP.
    xa = body_q[a_tcp_body_idx]
    pa = wp.transform_get_translation(xa)
    qa = wp.transform_get_rotation(xa)
    starts[3] = pa
    ends[3] = pa + wp.quat_rotate(qa, wp.vec3(axis_len, 0.0, 0.0))
    starts[4] = pa
    ends[4] = pa + wp.quat_rotate(qa, wp.vec3(0.0, axis_len, 0.0))
    starts[5] = pa
    ends[5] = pa + wp.quat_rotate(qa, wp.vec3(0.0, 0.0, axis_len))

    # Robot B TCP.
    xb = body_q[b_tcp_body_idx]
    pb = wp.transform_get_translation(xb)
    qb = wp.transform_get_rotation(xb)
    starts[6] = pb
    ends[6] = pb + wp.quat_rotate(qb, wp.vec3(axis_len, 0.0, 0.0))
    starts[7] = pb
    ends[7] = pb + wp.quat_rotate(qb, wp.vec3(0.0, axis_len, 0.0))
    starts[8] = pb
    ends[8] = pb + wp.quat_rotate(qb, wp.vec3(0.0, 0.0, axis_len))

    # Factory-equivalent TCP on Robot A.
    xha = body_q[a_hand_body_idx]
    xfa = wp.transform_multiply(xha, factory_offset)
    pfa = wp.transform_get_translation(xfa)
    qfa = wp.transform_get_rotation(xfa)
    starts[9] = pfa
    ends[9] = pfa + wp.quat_rotate(qfa, wp.vec3(axis_len, 0.0, 0.0))
    starts[10] = pfa
    ends[10] = pfa + wp.quat_rotate(qfa, wp.vec3(0.0, axis_len, 0.0))
    starts[11] = pfa
    ends[11] = pfa + wp.quat_rotate(qfa, wp.vec3(0.0, 0.0, axis_len))

    # Factory-equivalent TCP on Robot B.
    xhb = body_q[b_hand_body_idx]
    xfb = wp.transform_multiply(xhb, factory_offset)
    pfb = wp.transform_get_translation(xfb)
    qfb = wp.transform_get_rotation(xfb)
    starts[12] = pfb
    ends[12] = pfb + wp.quat_rotate(qfb, wp.vec3(axis_len, 0.0, 0.0))
    starts[13] = pfb
    ends[13] = pfb + wp.quat_rotate(qfb, wp.vec3(0.0, axis_len, 0.0))
    starts[14] = pfb
    ends[14] = pfb + wp.quat_rotate(qfb, wp.vec3(0.0, 0.0, axis_len))


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.test_mode = args.test

        urdf_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"

        builder = newton.ModelBuilder()
        # Robot A: always the Newton URDF, anchored at ROBOT_A_BASE.
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(ROBOT_A_BASE, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        # Robot B is always IsaacLab Factory's franka_mimic.usd.
        if not os.path.exists(_FACTORY_USD):
            raise SystemExit(
                f"[compare] ERROR: Factory USD not found at\n"
                f"          {_FACTORY_USD}\n"
                f"\n"
                f"This example loads IsaacLab's franka_mimic.usd as Robot B.\n"
                f"The file is not committed to this repo (NVIDIA Isaac Sim\n"
                f"license). Copy it into the expected path:\n"
                f"\n"
                f"   cp /tmp/Assets/Isaac/<version>/Isaac/IsaacLab/Factory/franka_mimic.usd \\\n"
                f"      {_FACTORY_USD}\n"
                f"\n"
                f"See newton/examples/assets/franka_mimic/README.md for the\n"
                f"full setup instructions."
            )
        print(f"[compare] Robot B: loading USD {_FACTORY_USD}")
        builder.add_usd(
            _FACTORY_USD,
            xform=wp.transform(ROBOT_B_BASE, wp.quat_identity()),
            only_load_enabled_rigid_bodies=True,
        )

        # Ground plane so it's visually obvious which way is up and where
        # left vs right is.
        builder.add_ground_plane()

        # Apply the same initial joint configuration to Robot A. Robot B may
        # have a different DOF layout (USD case) - we attempt to apply the
        # same arm/gripper q to its first 9 DOFs and warn if the count differs.
        builder.joint_q[:N_ARM_DOFS] = list(INIT_ARM_Q)
        builder.joint_q[N_ARM_DOFS:N_DOFS_PER_ROBOT] = list(INIT_FINGER_Q)
        n_dofs_total = len(builder.joint_q)
        # Robot B start index in the flat joint_q is whatever Robot A used.
        robot_b_q_start = N_DOFS_PER_ROBOT
        if n_dofs_total >= robot_b_q_start + N_DOFS_PER_ROBOT:
            builder.joint_q[robot_b_q_start : robot_b_q_start + N_ARM_DOFS] = list(INIT_ARM_Q)
            builder.joint_q[robot_b_q_start + N_ARM_DOFS : robot_b_q_start + N_DOFS_PER_ROBOT] = list(INIT_FINGER_Q)
        else:
            print(
                f"[compare] WARN: builder has {n_dofs_total} total DOFs - "
                f"Robot B doesn't have the expected {N_DOFS_PER_ROBOT} DOFs. "
                "Joint slider mirror may not align correctly."
            )

        self.model = builder.finalize()

        # Resolve Robot A indexing strictly (we control its asset).
        a_ee_label = builder.body_label[EE_BODY_OFFSET]
        assert a_ee_label.endswith("/fr3_hand_tcp"), f"Robot A EE label: {a_ee_label!r}"
        self.a_tcp_body_idx = EE_BODY_OFFSET
        self.a_hand_body_idx = HAND_BODY_OFFSET

        # Resolve Robot B's TCP body by name pattern - works for both the
        # Newton URDF (fr3_hand_tcp) and Factory's USD (panda_fingertip_centered).
        # Priority order: scan all bodies for the most-preferred candidate
        # first, only falling back if nothing matches.
        def _find_robot_b_body(candidates: tuple[str, ...], purpose: str) -> int:
            for cand in candidates:
                for body_idx in range(N_BODIES_PER_ROBOT, len(builder.body_label)):
                    label = builder.body_label[body_idx]
                    if label.endswith(f"/{cand}") or label == cand:
                        print(f"[compare] Robot B {purpose}: idx={body_idx}, label={label!r}")
                        return body_idx
            print(f"[compare] WARN: no body matched {candidates} for Robot B {purpose}")
            return -1

        self.b_tcp_body_idx = _find_robot_b_body(
            ("fr3_hand_tcp", "panda_fingertip_centered", "panda_hand"), "TCP body"
        )
        if self.b_tcp_body_idx < 0:
            self.b_tcp_body_idx = N_BODIES_PER_ROBOT + EE_BODY_OFFSET
        # Robot B's hand body is the parent on which the magenta "Factory
        # equivalent" frame is hung. Same name patterns as Newton's
        # fr3_hand / Factory's panda_hand.
        self.b_hand_body_idx = _find_robot_b_body(
            ("fr3_hand", "panda_hand"), "hand body"
        )
        if self.b_hand_body_idx < 0:
            self.b_hand_body_idx = N_BODIES_PER_ROBOT + HAND_BODY_OFFSET

        # Per-link comparison table: each row pairs the same kinematic frame
        # on Robot A and Robot B by name. Newton uses "fr3_*" naming, the
        # Factory USD uses "panda_*". The pair is included only if both
        # bodies are present in the scene.
        self._link_pairs = self._build_link_pairs(builder.body_label)
        for short_name, a_idx, b_idx in self._link_pairs:
            print(f"[compare] link pair {short_name:>10s}: A_idx={a_idx}, B_idx={b_idx}")

        # Body labels for the per-link delta readout.
        self._body_labels = list(builder.body_label)

        # State / FK only - no solver, no contacts. The viewer just shows
        # what FK produces from the current joint_q.
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # Debug-frame line buffers: 15 segments (3 axes * 5 frames).
        # See update_compare_frame_lines_kernel for the segment layout.
        n_segments = 15
        self._dbg_starts = wp.zeros(n_segments, dtype=wp.vec3, device=self.model.device)
        self._dbg_ends = wp.zeros(n_segments, dtype=wp.vec3, device=self.model.device)
        colors_host = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),  # 0..2  world axes
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),  # 3..5  Robot A TCP
            (0.4, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 0.4),  # 6..8  Robot B TCP
            (1.0, 0.0, 1.0), (1.0, 0.5, 0.0), (0.0, 1.0, 1.0),  # 9..11 Factory-eq on A
            (1.0, 0.0, 1.0), (1.0, 0.5, 0.0), (0.0, 1.0, 1.0),  # 12..14 Factory-eq on B
        ]
        self._dbg_colors = wp.array(colors_host, dtype=wp.vec3, device=self.model.device)
        self._show_debug_frames = True
        self._debug_axis_len = 0.05  # m
        self._world_axis_len = 0.20  # m -- longer so the world frame is obvious

        # Host-side joint state. The "mirror" mode keeps both robots in
        # sync; "independent" mode lets the user perturb each separately.
        self._gui_arm_q = list(INIT_ARM_Q)         # shared in mirror mode
        self._gui_finger_q = list(INIT_FINGER_Q)
        self._mirror = True
        self._gui_arm_q_b = list(INIT_ARM_Q)       # only used if not mirroring
        self._gui_finger_q_b = list(INIT_FINGER_Q)
        self._joint_state_dirty = True

        # Tunable Factory-equivalent TCP offset relative to fr3_hand. Default
        # ~0.1121 m in z is what the IsaacLab probe (scripts/probe_fingertip.py)
        # measured for ``panda_fingertip_centered`` relative to ``panda_hand``
        # at the Franka init pose. The Newton URDF's own ``fr3_hand_tcp`` is
        # 0.1035 m below ``fr3_hand``, so the magenta frame should sit ~9 mm
        # below Robot A's fr3_hand_tcp. Adjust via sliders to verify.
        self._factory_offset_pos = [0.0, 0.0, 0.1121]
        self._factory_offset_rpy = [0.0, 0.0, 0.0]

        # Cached per-link delta readout values, refreshed every N frames.
        self._delta_refresh_period = 10
        self._delta_top_pairs: list[tuple[str, float, float]] = []  # (name, dpos_mm, drot_deg)

        # Viewer setup.
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "renderer"):
            # Camera framing the robots from the front: positive Y is "back",
            # so we sit at y=-2.35 looking forward (yaw=90 deg, pitch=-3.4 deg)
            # toward both arms with a 45 deg FOV.
            self.viewer.set_camera(wp.vec3(0.05, -2.35, 0.6), -3.4, 90.0)
            self.viewer.camera.fov = 45.0
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self._frame = 0

    # ------------------------------------------------------------------
    # Joint state management
    # ------------------------------------------------------------------

    def _flush_joint_state_to_model(self) -> None:
        """Push host-side slider values into model.joint_q and refresh FK.

        When Robot B has a different DOF count (USD case), only the first
        ``min(robot_b_dofs, 9)`` of its DOFs receive the slider values; any
        extra DOFs in Robot B keep whatever default the asset shipped with.
        """
        if not self._joint_state_dirty:
            return
        if self._mirror:
            arm_b = self._gui_arm_q
            finger_b = self._gui_finger_q
        else:
            arm_b = self._gui_arm_q_b
            finger_b = self._gui_finger_q_b

        # Pull the current full host vector so we don't disturb DOFs we
        # don't know about (e.g. extra USD-side joints).
        full = self.model.joint_q.numpy().tolist()
        # Robot A: first 9 DOFs.
        for i in range(N_ARM_DOFS):
            if i < len(full):
                full[i] = float(self._gui_arm_q[i])
        for i in range(N_FINGER_DOFS):
            j = N_ARM_DOFS + i
            if j < len(full):
                full[j] = float(self._gui_finger_q[i])
        # Robot B: starting at index 9.
        for i in range(N_ARM_DOFS):
            j = N_DOFS_PER_ROBOT + i
            if j < len(full):
                full[j] = float(arm_b[i])
        for i in range(N_FINGER_DOFS):
            j = N_DOFS_PER_ROBOT + N_ARM_DOFS + i
            if j < len(full):
                full[j] = float(finger_b[i])
        self.model.joint_q.assign(full)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self._joint_state_dirty = False

    # ------------------------------------------------------------------
    # Sim loop (no dynamics; just FK + render)
    # ------------------------------------------------------------------

    def step(self) -> None:
        self._flush_joint_state_to_model()
        self.sim_time += self.frame_dt
        if self._frame % self._delta_refresh_period == 0:
            self._refresh_link_deltas()
        self._frame += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self._show_debug_frames:
            # Build the factory_offset transform from the current GUI values.
            roll, pitch, yaw = self._factory_offset_rpy
            q_off = wp.quat_rpy(float(roll), float(pitch), float(yaw))
            p_off = wp.vec3(*self._factory_offset_pos)
            xform_off = wp.transform(p_off, q_off)
            wp.launch(
                update_compare_frame_lines_kernel,
                dim=1,
                inputs=[
                    self.state.body_q,
                    self.a_tcp_body_idx,
                    self.b_tcp_body_idx,
                    self.a_hand_body_idx,
                    self.b_hand_body_idx,
                    xform_off,
                    float(self._world_axis_len),
                    float(self._debug_axis_len),
                ],
                outputs=[self._dbg_starts, self._dbg_ends],
                device=self.model.device,
            )
            self.viewer.log_lines("/compare_frames", self._dbg_starts, self._dbg_ends, self._dbg_colors)
        else:
            self.viewer.log_lines("/compare_frames", None, None, None)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _build_link_pairs(body_labels: list[str]) -> list[tuple[str, int, int]]:
        """Pair Robot-A and Robot-B bodies by canonical kinematic name.

        Returns a list of ``(display_name, a_body_idx, b_body_idx)`` for the
        Panda links that exist on both sides. Newton uses "fr3_*" naming,
        the Factory USD uses "panda_*". A pair is included only if a match
        was found for both robots.
        """
        # Canonical link order. First entry of each tuple is Newton/URDF
        # naming, second is Factory/USD naming.
        canonical = [
            ("link0", "fr3_link0", "panda_link0"),
            ("link1", "fr3_link1", "panda_link1"),
            ("link2", "fr3_link2", "panda_link2"),
            ("link3", "fr3_link3", "panda_link3"),
            ("link4", "fr3_link4", "panda_link4"),
            ("link5", "fr3_link5", "panda_link5"),
            ("link6", "fr3_link6", "panda_link6"),
            ("link7", "fr3_link7", "panda_link7"),
            ("hand", "fr3_hand", "panda_hand"),
            ("leftfinger", "fr3_leftfinger", "panda_leftfinger"),
            ("rightfinger", "fr3_rightfinger", "panda_rightfinger"),
            ("tcp", "fr3_hand_tcp", "panda_fingertip_centered"),
        ]

        def find(prefix_a_or_b: str, robot_range: range) -> int:
            for body_idx in robot_range:
                label = body_labels[body_idx]
                if label.endswith(f"/{prefix_a_or_b}") or label == prefix_a_or_b:
                    return body_idx
            return -1

        a_range = range(0, N_BODIES_PER_ROBOT)
        b_range = range(N_BODIES_PER_ROBOT, len(body_labels))
        pairs: list[tuple[str, int, int]] = []
        for short, a_name, b_name in canonical:
            a_idx = find(a_name, a_range)
            b_idx = find(b_name, b_range)
            if a_idx >= 0 and b_idx >= 0:
                pairs.append((short, a_idx, b_idx))
        return pairs

    def _refresh_link_deltas(self) -> None:
        """Pull body_q to host and compute per-link world positions and
        position/orientation deltas between the canonical Panda frames on
        Robot A vs Robot B. Pairs come from ``self._link_pairs`` so the
        readout is stable across frames and independent of body indexing
        differences between URDF and USD.

        Stores per-row tuples ``(name, dx_mm, dy_mm, dz_mm, dpos_mm, drot_deg,
        a_pos_xyz, b_pos_local)`` where ``b_pos_local`` is Robot B's world
        position with the base offset subtracted so it can be compared
        directly to Robot A's world position.
        """
        if not self._link_pairs:
            self._delta_top_pairs = []
            return
        body_q = self.state.body_q.numpy()
        from math import acos, degrees  # noqa: PLC0415

        base_dx = float(ROBOT_B_BASE[0]) - float(ROBOT_A_BASE[0])
        base_dy = float(ROBOT_B_BASE[1]) - float(ROBOT_A_BASE[1])
        base_dz = float(ROBOT_B_BASE[2]) - float(ROBOT_A_BASE[2])

        results = []
        for short_name, a_idx, b_idx in self._link_pairs:
            ax = body_q[a_idx]
            bx = body_q[b_idx]
            a_pos = (float(ax[0]), float(ax[1]), float(ax[2]))
            b_pos_local = (float(bx[0]) - base_dx, float(bx[1]) - base_dy, float(bx[2]) - base_dz)
            dx = b_pos_local[0] - a_pos[0]
            dy = b_pos_local[1] - a_pos[1]
            dz = b_pos_local[2] - a_pos[2]
            dpos_mm = ((dx * dx + dy * dy + dz * dz) ** 0.5) * 1000.0
            dot = abs(
                float(ax[3]) * float(bx[3])
                + float(ax[4]) * float(bx[4])
                + float(ax[5]) * float(bx[5])
                + float(ax[6]) * float(bx[6])
            )
            dot = min(1.0, max(-1.0, dot))
            drot_deg = degrees(2.0 * acos(dot))
            results.append(
                (short_name, dx * 1000.0, dy * 1000.0, dz * 1000.0, dpos_mm, drot_deg, a_pos, b_pos_local)
            )
        self._delta_top_pairs = results

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    @staticmethod
    def _input_and_slider_float(
        imgui, label: str, value: float, vmin: float, vmax: float, fmt: str = "%.4f"
    ) -> tuple[bool, float]:
        """Render an input field + slider on the same row sharing one value.

        Returns (changed, new_value). Either the input or the slider can mutate
        the value; whichever changed last in this frame wins. The label is
        shown to the right of the slider; the input has a hidden ID derived
        from the label so layout stays compact.
        """
        imgui.set_next_item_width(70)
        changed_in, v_in = imgui.input_float(f"##{label}_in", float(value), 0.0, 0.0, fmt)
        imgui.same_line()
        imgui.set_next_item_width(180)
        changed_sl, v_sl = imgui.slider_float(label, float(value), vmin, vmax)
        if changed_sl:
            return True, float(v_sl)
        if changed_in:
            return True, float(v_in)
        return False, float(value)

    def render_ui(self, imgui) -> None:
        imgui.separator()
        imgui.text("Side-by-side Panda comparison")
        imgui.separator()

        # Mirror toggle.
        changed, val = imgui.checkbox("Mirror joint angles", self._mirror)
        if changed:
            self._mirror = bool(val)
            if self._mirror:
                # Snap robot B back to robot A's pose.
                self._gui_arm_q_b = list(self._gui_arm_q)
                self._gui_finger_q_b = list(self._gui_finger_q)
            self._joint_state_dirty = True

        # Robot A arm sliders. Each row: input field (type exact rad value)
        # + slider (drag to scrub). Both write into the same _gui_arm_q[i].
        imgui.separator()
        imgui.text("Arm joint angles [rad]" + ("  (driving both)" if self._mirror else "  (Robot A)"))
        for i in range(N_ARM_DOFS):
            changed, v = self._input_and_slider_float(
                imgui, f"q{i}", float(self._gui_arm_q[i]), -3.1416, 3.1416
            )
            if changed:
                self._gui_arm_q[i] = float(v)
                if self._mirror:
                    self._gui_arm_q_b[i] = float(v)
                self._joint_state_dirty = True

        # Robot B arm sliders (only when independent).
        if not self._mirror:
            imgui.separator()
            imgui.text("Arm joint angles [rad]  (Robot B)")
            for i in range(N_ARM_DOFS):
                changed, v = self._input_and_slider_float(
                    imgui, f"qB{i}", float(self._gui_arm_q_b[i]), -3.1416, 3.1416
                )
                if changed:
                    self._gui_arm_q_b[i] = float(v)
                    self._joint_state_dirty = True

        # Quick reset to init pose.
        imgui.separator()
        if imgui.button("Reset to INIT_ARM_Q"):
            self._gui_arm_q = list(INIT_ARM_Q)
            self._gui_arm_q_b = list(INIT_ARM_Q)
            self._gui_finger_q = list(INIT_FINGER_Q)
            self._gui_finger_q_b = list(INIT_FINGER_Q)
            self._joint_state_dirty = True
        imgui.same_line()
        if imgui.button("Zero all joints"):
            self._gui_arm_q = [0.0] * N_ARM_DOFS
            self._gui_arm_q_b = [0.0] * N_ARM_DOFS
            self._joint_state_dirty = True

        # Factory TCP offset sliders (relative to fr3_hand). Same input + slider
        # treatment so you can type 0.1121 directly.
        imgui.separator()
        imgui.text("Factory TCP offset on both robots (relative to hand link)")
        labels = ("offset X [m]", "offset Y [m]", "offset Z [m]")
        for i, lbl in enumerate(labels):
            changed, v = self._input_and_slider_float(
                imgui, lbl, float(self._factory_offset_pos[i]), -0.30, 0.30, fmt="%.5f"
            )
            if changed:
                self._factory_offset_pos[i] = float(v)
        labels_rpy = ("offset roll [rad]", "offset pitch [rad]", "offset yaw [rad]")
        for i, lbl in enumerate(labels_rpy):
            changed, v = self._input_and_slider_float(
                imgui, lbl, float(self._factory_offset_rpy[i]), -3.1416, 3.1416
            )
            if changed:
                self._factory_offset_rpy[i] = float(v)

        # Visualization toggles.
        imgui.separator()
        changed, val = imgui.checkbox("Show debug frames", self._show_debug_frames)
        if changed:
            self._show_debug_frames = bool(val)
        changed, v = self._input_and_slider_float(
            imgui, "Axis length [m]", float(self._debug_axis_len), 0.01, 0.30
        )
        if changed:
            self._debug_axis_len = float(v)

        # Per-link world positions on Robot A and Robot B (re-based) plus
        # per-axis position deltas. Wrapped in a child region with both
        # horizontal and vertical scrollbars so the wide table fits even
        # when the side panel is narrow.
        imgui.separator()
        imgui.text("Per-link world positions  (B re-based to A):")
        child_flags = int(imgui.ChildFlags_.borders)
        window_flags = int(imgui.WindowFlags_.horizontal_scrollbar)
        # Height set to roughly fit the 12 canonical link rows + 2 header rows.
        if imgui.begin_child(
            "compare_link_table",
            imgui.ImVec2(0, 230),
            child_flags,
            window_flags,
        ):
            imgui.text(
                f"  {'link':>11s}  "
                f"{'Ax':>7s} {'Ay':>7s} {'Az':>7s}  "
                f"{'Bx':>7s} {'By':>7s} {'Bz':>7s}  "
                f"{'dx':>6s} {'dy':>6s} {'dz':>6s} {'|d|':>6s} {'drot':>6s}"
            )
            imgui.text(
                f"  {'':>11s}  "
                f"{'[m]':>7s} {'[m]':>7s} {'[m]':>7s}  "
                f"{'[m]':>7s} {'[m]':>7s} {'[m]':>7s}  "
                f"{'[mm]':>6s} {'[mm]':>6s} {'[mm]':>6s} {'[mm]':>6s} {'[deg]':>6s}"
            )
            for row in self._delta_top_pairs:
                name, dx_mm, dy_mm, dz_mm, dpos_mm, drot_deg, a_pos, b_pos = row
                imgui.text(
                    f"  {name:>11s}  "
                    f"{a_pos[0]:+7.3f} {a_pos[1]:+7.3f} {a_pos[2]:+7.3f}  "
                    f"{b_pos[0]:+7.3f} {b_pos[1]:+7.3f} {b_pos[2]:+7.3f}  "
                    f"{dx_mm:+6.1f} {dy_mm:+6.1f} {dz_mm:+6.1f} "
                    f"{dpos_mm:6.1f} {drot_deg:6.2f}"
                )
        imgui.end_child()
        imgui.text(f"frame: {self._frame}")

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """Sanity check that the kinematic chain (link0..link7) agrees
        between Robot A and Robot B at INIT_ARM_Q.

        The TCP and finger frames are intentionally excluded - Newton's
        URDF and IsaacLab's USD define those at slightly different points
        (gripper-opening center vs fingertip tips), and that real
        difference would otherwise trip this assertion.
        Tolerances reflect float32 FK noise: ~50 microns / ~0.1 degrees.
        """
        if not self.test_mode:
            return
        # Force a refresh in case the periodic refresh hasn't run yet.
        self._refresh_link_deltas()
        # Tuple layout: (name, dx, dy, dz, dpos_mm, drot_deg, a_pos, b_pos).
        chain_rows = [r for r in self._delta_top_pairs if r[0].startswith("link")]
        worst_pos = max((r[4] for r in chain_rows), default=0.0)
        worst_rot = max((r[5] for r in chain_rows), default=0.0)
        print(
            f"[compare-test] worst chain delta_pos = {worst_pos:.4f} mm, "
            f"worst chain delta_rot = {worst_rot:.4f} deg",
            flush=True,
        )
        assert worst_pos < 0.5, (  # 0.5 mm tolerates URDF-vs-USD float drift
            f"At INIT_ARM_Q, worst link0..link7 delta_pos = {worst_pos:.4f} mm. "
            "Kinematic chain frames should match between Robot A and Robot B."
        )
        assert worst_rot < 0.5, (  # 0.5 deg
            f"At INIT_ARM_Q, worst link0..link7 delta_quat = {worst_rot:.4f} deg."
        )

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=60)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
