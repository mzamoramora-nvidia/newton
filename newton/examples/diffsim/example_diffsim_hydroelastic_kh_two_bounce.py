# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Diffsim Hydroelastic KH Two Bounce
#
# Tunes the hydroelastic stiffness coefficient kh for a cube that bounces off
# a hydroelastic floor and then a hydroelastic wall to reach a target. The
# viewer shows the hydroelastic iso-surface and the contact force/torque
# wrench acting on the cube COM.
#
# Command: python -m newton.examples diffsim_hydroelastic_kh_two_bounce
#
###########################################################################

import numpy as np
import warp as wp
import warp.optim

import newton
import newton.examples
from newton.geometry import HydroelasticSDF
from newton.utils import bourke_color_map


@wp.kernel
def assign_uniform_kh(
    log_kh: wp.array[wp.float32],
    shape_indices: wp.array[wp.int32],
    shape_kh: wp.array[wp.float32],
):
    tid = wp.tid()
    kh = wp.exp(log_kh[0])
    shape_kh[shape_indices[tid]] = kh


@wp.kernel
def position_loss_kernel(
    body_q: wp.array[wp.transform],
    body_index: int,
    target: wp.vec3,
    final_pos: wp.array[wp.vec3],
    loss: wp.array[wp.float32],
):
    p = wp.transform_get_translation(body_q[body_index])
    err = p - target
    final_pos[0] = p
    loss[0] = wp.dot(err, err)


@wp.kernel
def clamp_kernel(lo: float, hi: float, x: wp.array[wp.float32]):
    tid = wp.tid()
    x[tid] = wp.clamp(x[tid], lo, hi)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.verbose = args.verbose

        self.frame = 0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = args.sim_steps
        self.sim_dt = 1.0 / 240.0
        self.render_interval = args.render_interval
        self.optimize_kh = True
        self.playback_slowdown = max(1, int(args.playback_slowdown))
        self.playback_step = 0
        self.playback_repeat = 0
        self.playback_rollout_dirty = False
        self.show_isosurface = args.show_isosurface
        self.show_wrench = args.show_wrench
        self.plot_history_size = args.plot_history_size

        self.cube_half = 0.1
        self.start_pos = (-0.55, 0.0, 0.45)
        self.start_vel = (2.3, 0.0, -2.0)
        self.start_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.22)
        self.start_ang_vel = (0.0, 2.5, 0.0)

        target_x = args.target_x
        if target_x is None:
            target_x = -0.10
        self.target = wp.vec3(target_x, args.target_y, args.target_z)

        self.train_iter = 0
        self.loss_history = []
        self.kh_history = []
        self.final_pos_history = []
        self.display_loss = 0.0
        self.initial_kh = float(args.initial_kh)

        self.lower_log_kh = np.log(args.min_kh)
        self.upper_log_kh = np.log(args.max_kh)
        self.log_kh = wp.array([np.log(self.initial_kh)], dtype=wp.float32, requires_grad=True)

        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.final_pos = wp.zeros(1, dtype=wp.vec3, requires_grad=True)

        self.model = self.create_model()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, angular_damping=0.0)
        self.control = self.model.control()
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.initial_state = self.model.state()
        self.initial_state.assign(self.states[0])

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=True,
                buffer_fraction=1.0,
                buffer_mult_iso=3,
                buffer_mult_contact=3,
            ),
            requires_grad=True,
        )
        self.step_contacts = [self.collision_pipeline.contacts() for _ in range(self.sim_steps)]
        self.render_contacts = self.collision_pipeline.contacts()

        self.optimizer = warp.optim.Adam([self.log_kh], lr=args.train_rate)

        self.viewer.set_model(self.model)
        self.viewer.show_hydro_contact_surface = self.show_isosurface
        self.viewer.set_camera(
            pos=wp.vec3(0, -2.5, 1.0),
            pitch=-15.0,
            yaw=90.0,
        )
        if hasattr(self.viewer, "_plot_history_size"):
            self.viewer._plot_history_size = max(self.viewer._plot_history_size, self.plot_history_size)

    def _current_kh(self):
        return float(np.exp(self.log_kh.numpy()[0]))

    def create_model(self):
        kh = self._current_kh()
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            sdf_max_resolution=32,
            sdf_narrow_band_range=(-0.05, 0.05),
            is_hydroelastic=True,
            gap=0.01,
            kh=kh,
            kd=0.0,
            kf=0.0,
            mu=0.0,
        )

        builder = newton.ModelBuilder(gravity=-9.81)
        self.floor_shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
            hx=0.9,
            hy=0.3,
            hz=0.05,
            cfg=shape_cfg,
        )
        self.wall_shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.45, 0.0, 0.35), wp.quat_identity()),
            hx=0.05,
            hy=0.3,
            hz=0.35,
            cfg=shape_cfg,
        )
        self.cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.start_pos), self.start_rot),
            label="two_bounce_hydroelastic_cube",
        )
        builder.joint_qd[0] = builder.body_qd[-1][0] = self.start_vel[0]
        builder.joint_qd[1] = builder.body_qd[-1][1] = self.start_vel[1]
        builder.joint_qd[2] = builder.body_qd[-1][2] = self.start_vel[2]
        builder.joint_qd[3] = builder.body_qd[-1][3] = self.start_ang_vel[0]
        builder.joint_qd[4] = builder.body_qd[-1][4] = self.start_ang_vel[1]
        builder.joint_qd[5] = builder.body_qd[-1][5] = self.start_ang_vel[2]
        self.cube_shape = builder.add_shape_box(
            body=self.cube_body,
            hx=self.cube_half,
            hy=self.cube_half,
            hz=self.cube_half,
            cfg=shape_cfg,
        )

        model = builder.finalize(requires_grad=True)
        self.shape_indices = wp.array(
            [self.floor_shape, self.wall_shape, self.cube_shape],
            dtype=wp.int32,
        )
        return model

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

    def forward(self):
        self.states[0].assign(self.initial_state)

        wp.launch(
            assign_uniform_kh,
            dim=self.shape_indices.shape[0],
            inputs=[self.log_kh, self.shape_indices],
            outputs=[self.model.shape_material_kh],
        )

        for sim_step in range(self.sim_steps):
            self.states[sim_step].clear_forces()
            contacts = self.step_contacts[sim_step]
            self.collision_pipeline.collide(self.states[sim_step], contacts)
            self.solver.step(
                self.states[sim_step],
                self.states[sim_step + 1],
                self.control,
                contacts,
                self.sim_dt,
            )

        wp.launch(
            position_loss_kernel,
            dim=1,
            inputs=[self.states[-1].body_q, self.cube_body, self.target],
            outputs=[self.final_pos, self.loss],
        )
        return self.loss

    def step(self):
        if not self.optimize_kh:
            if self.playback_rollout_dirty:
                self.forward()
                self.display_loss = float(self.loss.numpy()[0])
                self.loss.zero_()
                self.final_pos.zero_()
                self.playback_rollout_dirty = False
            return

        self.forward_backward()

        kh = self._current_kh()
        loss = float(self.loss.numpy()[0])
        final_pos = self.final_pos.numpy()[0]
        grad = float(self.log_kh.grad.numpy()[0])
        if self.verbose:
            print(
                f"Train iter: {self.train_iter} Loss: {loss:.6e} "
                f"Final: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}) "
                f"kh: {kh:.6e} grad(log_kh): {grad:.6e}"
            )

        self.optimizer.step([self.log_kh.grad])
        wp.launch(clamp_kernel, dim=1, inputs=[self.lower_log_kh, self.upper_log_kh], outputs=[self.log_kh])

        self.loss_history.append(loss)
        self.kh_history.append(kh)
        self.final_pos_history.append(final_pos)
        self.display_loss = loss
        self._log_training_plots(loss, kh)

        self.tape.zero()
        self.loss.zero_()
        self.final_pos.zero_()

        self.train_iter += 1
        self.playback_step = 0
        self.playback_repeat = 0
        self.playback_rollout_dirty = True

    def _rollout_hit_shapes(self):
        floor_hit = False
        wall_hit = False
        for contacts in self.step_contacts:
            count = int(contacts.rigid_contact_count.numpy()[0])
            if count == 0:
                continue
            pairs = np.column_stack(
                (
                    contacts.rigid_contact_shape0.numpy()[:count],
                    contacts.rigid_contact_shape1.numpy()[:count],
                )
            )
            floor_hit = floor_hit or np.any(
                ((pairs[:, 0] == self.floor_shape) & (pairs[:, 1] == self.cube_shape))
                | ((pairs[:, 1] == self.floor_shape) & (pairs[:, 0] == self.cube_shape))
            )
            wall_hit = wall_hit or np.any(
                ((pairs[:, 0] == self.wall_shape) & (pairs[:, 1] == self.cube_shape))
                | ((pairs[:, 1] == self.wall_shape) & (pairs[:, 0] == self.cube_shape))
            )
            if floor_hit and wall_hit:
                return True, True
        return floor_hit, wall_hit

    def test_final(self):
        if len(self.loss_history) < 2:
            raise AssertionError("Expected at least two training iterations")
        best_loss = min(self.loss_history)
        if not best_loss < self.loss_history[0]:
            raise AssertionError(f"Expected loss to decrease, got {self.loss_history}")
        if len(self.loss_history) >= 20 and not (best_loss < 1.0e-4 or best_loss < 0.25 * self.loss_history[0]):
            raise AssertionError(f"Expected long run to converge, got {self.loss_history}")
        floor_hit, wall_hit = self._rollout_hit_shapes()
        if not floor_hit or not wall_hit:
            raise AssertionError(
                f"Expected floor and wall hydroelastic contacts, got floor={floor_hit}, wall={wall_hit}"
            )
        qd = np.array([state.body_qd.numpy()[self.cube_body, :3] for state in self.states])
        if np.max(qd[:, 2]) <= 0.2:
            raise AssertionError("Expected upward velocity after floor bounce")
        if np.min(qd[:, 0]) >= -0.2:
            raise AssertionError("Expected negative x velocity after wall bounce")
        if abs(self.kh_history[-1] - self.kh_history[0]) <= 1.0:
            raise AssertionError(f"Expected kh to be updated, got {self.kh_history}")

    def _log_training_plots(self, loss, kh):
        # Log once per training iteration so the plot window keeps optimizer history,
        # not repeated samples from every rendered rollout state.
        self.viewer.log_scalar("/loss", loss)
        self.viewer.log_scalar("/log10_loss", np.log10(max(loss, 1.0e-12)))
        self.viewer.log_scalar("/kh_1e6", kh / 1.0e6)

    def _log_wrench_arrows(self, state):
        if not self.show_wrench:
            self.viewer.log_arrows("/cube_contact_force", None, None, None)
            self.viewer.log_arrows("/cube_contact_torque", None, None, None)
            return

        body_q = state.body_q.numpy()[self.cube_body]
        com = np.array(body_q[:3], dtype=np.float32)
        wrench = state.body_f.numpy()[self.cube_body]
        force = np.array(wrench[:3], dtype=np.float32)
        torque = np.array(wrench[3:6], dtype=np.float32)

        self._log_single_arrow(
            "/cube_contact_force",
            com,
            force,
            scale=2.5e-5,
            max_len=0.28,
            threshold=1.0e-4,
            color=(1.0, 0.85, 0.05),
        )
        self._log_single_arrow(
            "/cube_contact_torque",
            com,
            torque,
            scale=2.0e-3,
            max_len=0.22,
            threshold=1.0e-5,
            color=(0.0, 0.8, 1.0),
        )

    def _log_single_arrow(self, name, start, vector, scale, max_len, threshold, color):
        if np.linalg.norm(vector) <= threshold:
            self.viewer.log_arrows(name, None, None, None)
            return

        arrow = vector * scale
        arrow_len = np.linalg.norm(arrow)
        if arrow_len > max_len:
            arrow *= max_len / arrow_len

        self.viewer.log_arrows(
            name,
            wp.array([start], dtype=wp.vec3),
            wp.array([start + arrow], dtype=wp.vec3),
            wp.array([color], dtype=wp.vec3),
        )

    def _render_state(self, state, traj_verts, loss, color):
        self.viewer.begin_frame(self.frame * self.frame_dt)
        self.viewer.log_state(state)
        self.collision_pipeline.collide(state, self.render_contacts)
        self.viewer.log_hydro_contact_surface(
            (
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                if self.collision_pipeline.hydroelastic_sdf is not None
                else None
            ),
            penetrating_only=False,
        )
        self._log_wrench_arrows(state)
        self.viewer.log_shapes(
            "/target",
            newton.GeoType.BOX,
            (0.08, 0.08, 0.08),
            wp.array([wp.transform(self.target, wp.quat_identity())], dtype=wp.transform),
            wp.array([wp.vec3(0.5, 0.0, 0.5)], dtype=wp.vec3),
        )
        self.viewer.log_lines(
            f"/traj_{max(self.train_iter - 1, 0)}",
            wp.array(traj_verts[:-1], dtype=wp.vec3),
            wp.array(traj_verts[1:], dtype=wp.vec3),
            color,
        )
        self.viewer.end_frame()
        self.frame += 1

    def _render_playback(self, traj_verts, loss, color):
        if self.playback_step >= len(self.states):
            self.playback_step = 0
            self.playback_repeat = 0

        self._render_state(self.states[self.playback_step], traj_verts, loss, color)

        self.playback_repeat += 1
        if self.playback_repeat >= self.playback_slowdown:
            self.playback_repeat = 0
            self.playback_step += 1

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return

        if self.optimize_kh and self.frame > 0 and self.train_iter % self.render_interval != 0:
            return

        traj_verts = [state.body_q.numpy()[self.cube_body, :3].tolist() for state in self.states]
        loss = self.display_loss
        color = (
            bourke_color_map(0.0, max(self.loss_history[0], 1.0e-9), loss)
            if self.loss_history
            else (1.0, 0.0, 0.0)
        )

        if not self.optimize_kh:
            self._render_playback(traj_verts, loss, color)
            return

        for state in self.states:
            self._render_state(state, traj_verts, loss, color)

    def gui(self, imgui):
        imgui.text_wrapped("Uncheck Optimize KH to replay the rollout and set a slowdown factor.")
        changed, self.optimize_kh = imgui.checkbox("Optimize KH", self.optimize_kh)
        if changed:
            self.playback_step = 0
            self.playback_repeat = 0
            self.playback_rollout_dirty = not self.optimize_kh

        if not self.optimize_kh:
            _changed, self.playback_slowdown = imgui.slider_int("Rollout Slowdown", self.playback_slowdown, 1, 20)

        changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_isosurface

        _changed, self.show_wrench = imgui.checkbox("Show Force/Torque", self.show_wrench)
        imgui.separator()
        imgui.text(f"initial kh: {self.initial_kh:.6e}")
        imgui.text(f"kh: {self._current_kh():.6e}")
        imgui.text(f"train iter: {self.train_iter}")
        if self.loss_history or self.display_loss > 0.0:
            imgui.text(f"loss: {self.display_loss:.6e}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--verbose", action="store_true", help="Print training diagnostics.")
        parser.add_argument("--sim-steps", type=int, default=140, help="Number of simulation steps per rollout.")
        parser.add_argument("--initial-kh", type=float, default=1.0e7, help="Initial hydroelastic stiffness.")
        parser.add_argument("--min-kh", type=float, default=1.0e3, help="Lower optimization bound for kh.")
        parser.add_argument("--max-kh", type=float, default=1.0e9, help="Upper optimization bound for kh.")
        parser.add_argument("--target-x", type=float, default=None, help="Target final x-position.")
        parser.add_argument("--target-y", type=float, default=0.0, help="Target final y-position.")
        parser.add_argument("--target-z", type=float, default=0.62, help="Target final height after floor/wall bounce.")
        parser.add_argument("--train-rate", type=float, default=0.04, help="Adam learning rate for log(kh).")
        parser.add_argument(
            "--render-interval", type=int, default=10, help="Render one trajectory every N train steps."
        )
        parser.add_argument(
            "--playback-slowdown", type=int, default=1, help="Repeat each rendered rollout frame N times."
        )
        parser.add_argument(
            "--show-isosurface", action="store_true", default=True, help="Show hydroelastic iso-surface."
        )
        parser.add_argument("--hide-isosurface", action="store_false", dest="show_isosurface")
        parser.add_argument(
            "--show-wrench", action="store_true", default=True, help="Show COM force and torque arrows."
        )
        parser.add_argument("--hide-wrench", action="store_false", dest="show_wrench")
        parser.add_argument(
            "--plot-history-size",
            type=int,
            default=1000,
            help="Number of optimization samples kept in the viewer plots.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
