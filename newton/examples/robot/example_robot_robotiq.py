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


class Example:
    def __init__(self, viewer, num_worlds=4):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        self.device = wp.get_device()

        # Path to Robotiq 2f85 gripper asset (requires mujoco_menagerie)
        robotiq_2f85_dir = "/home/mzamoramora/build_playground/mujoco_menagerie/robotiq_2f85"
        robotiq_2f85_path = f"{robotiq_2f85_dir}/2f85.xml"

        # When loading from the mujoco_menagerie/robotiq_2f85_v4 folder, the resulting inertias that are computed by Newton result in
        # small negative values (-2e-20), which lead to a parsing error being triggered by the mujoco spec.
        # Setting ignore_inertial_definitions=False when adding the mjcf model does not trigger the parsing error.
        # However, the meshes are not shown in the viewer.
        # The main difference between the two models is that v4 sets coef=0.485 in the joints of the fixed tendon to make sure
        # that the finger tips can touch each other when closing.

        # Use MuJoCo to resolve includes and flatten the XML
        # This is needed because Newton's MJCF parser doesn't handle <include> tags
        print("Resolving MJCF includes...")
        mj_spec = mujoco.MjSpec.from_file(robotiq_2f85_path)
        flattened_xml = mj_spec.to_xml()

        # Save flattened XML to the asset directory so relative mesh paths resolve correctly
        flattened_path = f"{robotiq_2f85_dir}/_flattened_for_newton.xml"
        with open(flattened_path, "w") as f:
            f.write(flattened_xml)

        # Build the Shadow Hand model
        robotiq_2f85 = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robotiq_2f85)

        quat = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), wp.pi) * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi / 2)
        xform = wp.transform(wp.vec3(0, 0, 0.385), quat)
        robotiq_2f85.add_mjcf(
            flattened_path,
            xform=xform,
            verbose=True,
        )

        # Clean up the temp file
        os.remove(flattened_path)

        #===============================================
        # Overriding values instead of creating a new asset.
        #===============================================      
        # solreflimit is converted to joint_limit_ke and joint_limit_kd.
        # so, we need to override the values here.
        robotiq_2f85.joint_limit_ke[:] = [10000.0] * robotiq_2f85.joint_dof_count
        robotiq_2f85.joint_limit_ke[2] = 2500.0
        robotiq_2f85.joint_limit_ke[6] = 2500.0

        robotiq_2f85.joint_limit_kd[:] = [100.0] * robotiq_2f85.joint_dof_count

        #===============================================      
        # Override tendon coefficients as in 2f85_v4.xml to make sure the finger tips can touch each other when closing.
        robotiq_2f85.custom_attributes["mujoco:tendon_coef"].values = [0.485, 0.485]

        # Stiffness, damping and spring ref for couplers (indexes 1 and 5)
        robotiq_2f85.custom_attributes["mujoco:dof_passive_stiffness"].values = {1: float(2.0), 5: float(2.0)}
        robotiq_2f85.custom_attributes["mujoco:dof_passive_damping"].values = {1: float(0.3), 5: float(0.3)}
        robotiq_2f85.custom_attributes["mujoco:dof_springref"].values = {1: float(30.0), 5: float(30.0)}
        #===============================================

        # Store joints per world for the kernel
        self.joints_per_world = robotiq_2f85.joint_count

        # Get tendon info and extract the first joint of each tendon
        tendon_joint_adr = robotiq_2f85.custom_attributes["mujoco:tendon_joint_adr"].values or {}
        tendon_joint_vals = robotiq_2f85.custom_attributes["mujoco:tendon_joint"].values or {}
        self.tendons_per_world = len(tendon_joint_adr)

        # Get the first joint index for each tendon (we'll animate this one, tendon couples the rest)
        tendon_first_joints = []
        for i in range(self.tendons_per_world):
            adr = tendon_joint_adr[i]
            first_joint = tendon_joint_vals[adr]
            tendon_first_joints.append(first_joint)
            joint_name = robotiq_2f85.joint_key[first_joint]
            print(f"Tendon {i}: first joint = {joint_name} (idx {first_joint})")

        self.tendon_first_joints_template = tendon_first_joints

        print(f"\nParsed {self.tendons_per_world} fixed tendons")

        # The actuated joint are right_driver_joint and left_driver_joint
        # and have dof indexes 0 and 4.
        # TODO: Check that we are parsing the joint params (armature, stiffness, etc) correctly.

        for i in [0, 4]:
            robotiq_2f85.joint_target_ke[i] = 20.0
            robotiq_2f85.joint_target_kd[i] = 1.0
            robotiq_2f85.joint_target_pos[i] = 0.0    



        self.table_height = 0.2
        self.cube_size = 0.05
        robotiq_2f85.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0, 0, 0.5*self.table_height)), hx=0.2, hy=0.2, hz=0.5*self.table_height)
        cube_body = robotiq_2f85.add_body(xform=wp.transform(wp.vec3(0, 0, self.table_height+0.5*self.cube_size)))
        robotiq_2f85.add_shape_box(body=cube_body, hx=0.5*self.cube_size, hy=0.5*self.cube_size, hz=0.5*self.cube_size)


        # Create main builder and replicate for multi-world
        builder = newton.ModelBuilder()
        builder.replicate(robotiq_2f85, self.num_worlds)

        builder.add_ground_plane()

        self.model = builder.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

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

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.collide(self.state_0)

        self.target_pos = 0.0
        self.joint_target_pos = wp.zeros_like(self.control.joint_target_pos)
        wp.copy(self.joint_target_pos, self.control.joint_target_pos)

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

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Set new targets (self.joint_target_pos) acquired from GUI
        wp.copy(self.control.joint_target_pos, self.joint_target_pos)

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
        print("Gripper simulation completed successfully")

    def gui(self, imgui):
        imgui.text("Gripper target")

        changed, value = imgui.slider_float("target_pos", self.target_pos, 0.0, 0.8, format="%.3f")
        if changed:
            self.target_pos = value
            # The actuated joint are right_driver_joint and left_driver_joint
            # and have dof indexes 0 and 4.
            # We set the same target for both joints for all worlds.
            joint_target_pos = self.joint_target_pos.reshape((self.num_worlds, -1)).numpy()
            joint_target_pos[:, 0] = value
            joint_target_pos[:, 4] = value
            # print(joint_target_pos)
            wp.copy(self.joint_target_pos, wp.array(joint_target_pos.flatten(), dtype=wp.float32))


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=4, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
