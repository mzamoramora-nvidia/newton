# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example USD Mixed Deformables
#
# Command: uv run -m newton.examples vbd_usd_deformables_mixed
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_USD_PATH = newton.examples.get_asset("deformables_random_scene.usda")


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_usd(args.usd_path)
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=10.0, mu=0.8, gap=0.002),
        )
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0
        self.model.soft_contact_mu = 0.6

        pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            contact_matching="latest",
        )
        self.contacts = self.model.contacts(collision_pipeline=pipeline)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.vbd_iterations,
            particle_enable_self_contact=not args.disable_self_contact,
            particle_self_contact_radius=0.01,
            particle_self_contact_margin=0.02,
            rigid_contact_history=True,
            rigid_avbd_contact_alpha=0.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.0, -2.0, 1.5),
            pitch=-15.0,
            yaw=120.0,
        )

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        assert np.isfinite(particle_q).all(), "Non-finite particle positions"
        assert np.isfinite(particle_qd).all(), "Non-finite particle velocities"
        assert np.linalg.norm(particle_q, axis=1).max() < 100.0, "Particle state escaped the scene"

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.isfinite(body_q).all(), "Non-finite cable body poses"
        assert np.isfinite(body_qd).all(), "Non-finite cable body velocities"
        assert np.linalg.norm(body_q[:, :3], axis=1).max() < 100.0, "Cable state escaped the scene"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--usd-path",
            type=str,
            default=DEFAULT_USD_PATH,
            help="USD scene containing curve, surface, and volume deformables.",
        )
        parser.add_argument(
            "--sim-substeps",
            type=int,
            default=4,
            help="VBD substeps per display frame.",
        )
        parser.add_argument(
            "--vbd-iterations",
            type=int,
            default=4,
            help="VBD iterations per simulation substep.",
        )
        parser.add_argument(
            "--disable-self-contact",
            action="store_true",
            help="Disable particle self-contact for a faster preview.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
