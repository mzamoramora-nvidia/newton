# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Robot Shadow Hand
#
# Shows how to set up a simulation of a Shadow Hand with fixed tendons.
# The Shadow Hand uses tendons to couple finger joints together
# (e.g., the two joints of each fingertip are coupled via a fixed tendon).
#
# This example demonstrates:
# - Loading a MuJoCo model with fixed tendons via MJCF
# - Using MjSpec to resolve MJCF includes before parsing
# - Multi-world simulation with tendon support
#
# Command: python -m newton.examples robot_shadow_hand --num-worlds 4
#
###########################################################################

import os

import mujoco
import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def move_tendon_joints(
    tendon_joint_indices: wp.array(dtype=wp.int32),  # Indices of joints in tendons (one per tendon)
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    sim_dt: float,
    tendons_per_world: int,
    joints_per_world: int,
    # outputs
    joint_target_pos: wp.array(dtype=wp.float32),
):
    world_id = wp.tid()
    t = sim_time[world_id]

    # Only animate the first joint of each tendon (the tendon will couple to the second)
    for i in range(tendons_per_world):
        tendon_idx = world_id * tendons_per_world + i
        joint_idx = tendon_joint_indices[tendon_idx]
        di = joint_qd_start[joint_idx]

        # Create a wave pattern - each finger moves with a phase offset
        target = wp.sin(t * 2.0 + float(i) * 0.5) * 0.4 + 0.6
        joint_target_pos[di] = wp.clamp(target, joint_limit_lower[di], joint_limit_upper[di])

    # update the sim time
    sim_time[world_id] += sim_dt


class Example:
    def __init__(self, viewer, num_worlds=4):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        self.device = wp.get_device()

        # Path to Shadow Hand asset (requires mujoco_menagerie)
        # shadow_hand_dir = "/home/adenzler/git/mujoco_menagerie/shadow_hand"
        shadow_hand_dir = "/home/mzamoramora/build_playground/mujoco_menagerie/robotiq_2f85"
        shadow_hand_path = f"{shadow_hand_dir}/2f85.xml"

        # When loading from the mujoco_menagerie/robotiq_2f85_v4 folder, the resulting inertias that are computed by Newton result in
        # small negative values (-2e-20), which lead to a parsing error being triggered by the mujoco spec.
        # Setting ignore_inertial_definitions=False when adding the mjcf model does not trigger the parsing error.
        # However, the meshes are not shown in the viewer.
        # The main difference between the two models is that v4 sets coef=0.485 in the joints of the fixed tendon to make sure
        # that the finger tips can touch each other when closing.

        # Use MuJoCo to resolve includes and flatten the XML
        # This is needed because Newton's MJCF parser doesn't handle <include> tags
        print("Resolving MJCF includes...")
        mj_spec = mujoco.MjSpec.from_file(shadow_hand_path)
        flattened_xml = mj_spec.to_xml()

        # Save flattened XML to the asset directory so relative mesh paths resolve correctly
        flattened_path = f"{shadow_hand_dir}/_flattened_for_newton.xml"
        with open(flattened_path, "w") as f:
            f.write(flattened_xml)

        # Build the Shadow Hand model
        shadow_hand = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(shadow_hand)

        shadow_hand.add_mjcf(flattened_path, verbose=True)

        # Clean up the temp file
        os.remove(flattened_path)

        # Store joints per world for the kernel
        self.joints_per_world = shadow_hand.joint_count

        # Get tendon info and extract the first joint of each tendon
        tendon_joint_adr = shadow_hand.custom_attributes["mujoco:tendon_joint_adr"].values or {}
        tendon_joint_vals = shadow_hand.custom_attributes["mujoco:tendon_joint"].values or {}
        self.tendons_per_world = len(tendon_joint_adr)

        # Get the first joint index for each tendon (we'll animate this one, tendon couples the rest)
        tendon_first_joints = []
        for i in range(self.tendons_per_world):
            adr = tendon_joint_adr[i]
            first_joint = tendon_joint_vals[adr]
            tendon_first_joints.append(first_joint)
            joint_name = shadow_hand.joint_key[first_joint]
            print(f"Tendon {i}: first joint = {joint_name} (idx {first_joint})")

        self.tendon_first_joints_template = tendon_first_joints

        print(f"\nParsed {self.tendons_per_world} fixed tendons")

        # Set joint targets and joint drive gains
        for i in range(shadow_hand.joint_dof_count):
            shadow_hand.joint_target_ke[i] = 50
            shadow_hand.joint_target_kd[i] = 2
            shadow_hand.joint_target_pos[i] = 0.5

        # Create main builder and replicate for multi-world
        builder = newton.ModelBuilder()
        builder.replicate(shadow_hand, self.num_worlds)

        builder.add_ground_plane()

        self.model = builder.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.world_time = wp.zeros(self.num_worlds, dtype=wp.float32)

        # Build array of first joint indices for each tendon across all worlds
        # Joint indices get offset by joints_per_world for each world during replicate
        all_tendon_joints = []
        for w in range(self.num_worlds):
            joint_offset = w * self.joints_per_world
            for j in self.tendon_first_joints_template:
                all_tendon_joints.append(j + joint_offset)
        self.tendon_joint_indices = wp.array(all_tendon_joints, dtype=wp.int32)

        # Print final tendon info from model
        print("\n=== Model Tendon Info ===")
        print(f"Tendons per world: {self.tendons_per_world}")
        print(f"Total tendons: {len(self.model.mujoco.tendon_world)}")

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=500,
            nconmax=300,
            iterations=50,
            ls_iterations=25,
            use_mujoco_cpu=False,
        )

        # Print MuJoCo tendon info
        print(f"\nMuJoCo model ntendon: {self.solver.mj_model.ntendon}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                move_tendon_joints,
                dim=self.num_worlds,
                inputs=[
                    self.tendon_joint_indices,
                    self.model.joint_qd_start,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.world_time,
                    self.sim_dt,
                    self.tendons_per_world,
                    self.joints_per_world,
                ],
                outputs=[self.control.joint_target_pos],
            )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Basic sanity check - bodies should not have NaN positions
        body_q = self.state_0.body_q.numpy()
        assert not np.any(np.isnan(body_q)), "Body positions contain NaN values"
        print("Shadow hand simulation completed successfully")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=4, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
