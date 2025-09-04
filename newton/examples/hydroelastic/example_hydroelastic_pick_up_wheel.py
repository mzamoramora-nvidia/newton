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
# Example Hydroelastic Pick Up Wheel
#
# Shows how to use Newton to simulate a wheel being picked up by a gripper.
#
# Before running the example, make sure to initialize the hydroelastic-assets
# submodule by running:
#
# git submodule update --init --recursive
#
# Commands: python -m newton.examples hydroelastic_pick_up_wheel
#           uv run -m newton.examples hydroelastic_pick_up_wheel
#
# Optional arguments:
# --viewer=usd --output-path hydroelastic_pick_up_wheel.usd --num-frames=2000
#
###########################################################################

from __future__ import annotations

import time

import numpy as np
import warp as wp

import newton
import newton._src.hydroelastic.imgui as hydroelastic_imgui
import newton._src.hydroelastic.isosurface as hydroelastic_isosurface
import newton._src.hydroelastic.loaders as hydroelastic_loaders
import newton._src.hydroelastic.render_utils as hydroelastic_render_utils
import newton._src.hydroelastic.utils as hydroelastic_utils
import newton._src.hydroelastic.wrenches as hydroelastic_wrenches
import newton.examples
import newton.utils

# wp.config.verbose = True
wp.config.verbose_warnings = True
# wp.config.mode = "debug" # "release"
# wp.config.cache_kernels = True
# wp.config.verify_cuda = True
# wp.config.verify_fp = True


class EditableVars:
    def __init__(self):
        # Drawing vars
        self.np_vertex_offset = wp.vec3(0.0, 0.2, 0.0)  # [0.0, 0.2, 0.0]
        self.proxy_of_max_tet_pairs = 80  # 150  # len(self.tet_pairs)

        self.render_isosurfaces_flag = False
        self.render_forces_flag = True
        # With a scale of 0.01, an object of 1kg that results in a force of 9.8N, will have an arrow of approx 0.01m = 1cm.
        # The maximum gripping force of Robotiq 2F-140 is 125N. With a scale of 0.001, the arrow will be 0.125m = 12.5cm.
        self.force_scale = 0.1  # 0.01
        # Enabling plotting will make the simulation very slow.
        # TODO: Figure out a way to replace matplotlib with ImGuiPlot via imgui-bundle.
        self.plot_flag = False


class Example:
    def __init__(self, viewer, verbose=False):
        # setup simulation parameters first
        self.fps = 200
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1000  # Use 100 substeps with fps = 100 to avoid slippage.
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose
        self.viewer = viewer
        self.time_start = time.perf_counter()

        self.use_imgui = True  # self.render_mode == RenderMode.OPENGL
        self.up_axis = newton.Axis.Z
        self.dirs = hydroelastic_utils.get_dirs(self.up_axis)
        self.device = wp.get_device()
        self.use_cuda_graph = True
        self.editable_vars = EditableVars()
        self.editable_vars.np_vertex_offset = 0.5 * self.dirs.up

        # Load model
        self.model = self.load_model()

        # Setup solver
        # self.solver = newton.solvers.SolverSemiImplicit(self.model, joint_attach_ke=20000, joint_attach_kd=1.0e2)
        self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=0.100, update_mass_matrix_interval=self.sim_substeps)
        # self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        # Get id for twist convention. (TODO: Change to enum instead of int)
        self.twist_convention = hydroelastic_utils.get_twist_convention(self.solver)
        self.twist_convention_wp = wp.int32(self.twist_convention)

        # Initialize simulation states, control and standard contacts.
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.005)

        # ==============================================================================================================
        self.joint_target = wp.array(self.control.joint_target.numpy(), dtype=wp.float32)

        # ==============================================================================================================
        # This is a hack to disable normal contacts for featherstone solver. Not sure if it works for other solvers.
        self.contacts.rigid_contact_max = 0

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ==============================================================================================================
        # Setup hydroelastic contacts
        self.contacts.use_hydroelastic_inside_solver = False
        self.contacts.isosurface = []
        self.contacts.num_isosurfaces = hydroelastic_loaders.init_isosurfaces(
            self.model.collision_pairs, self.contacts.isosurface, self.model.hydro_mesh, self.device
        )

        # ==============================================================================================================
        # Perform fake step to initialize the solver.
        # This is useful for the Featherstone solver in particular, to populate stuff like body_v_s.
        self.body_q_inv_mat = wp.array(shape=(self.model.body_count,), dtype=wp.mat44, device=self.device)
        self.compute_contact_surfaces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

        # TODO: Find a better place to store this body_f.
        self.body_f = wp.zeros_like(self.state_1.body_f)

        # ==============================================================================================================
        # Setup viewer
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.setup_imgui()
        elif isinstance(self.viewer, newton.viewer.ViewerUSD):
            self.viewer.fps = self.fps
            self.viewer.stage.SetFramesPerSecond(self.fps)

        # ==============================================================================================================
        # Setup data history and figures for plotting
        self.data_history = []
        self.figs = []
        self.axs = []
        self.append_to_data_history()

        # ==============================================================================================================
        # Capture graph
        self.capture()

        # ==============================================================================================================
        print("Done init")

    def load_model(self):
        # ==============================================================================================================
        # Loading of the meshes should be moved somewhere else (e.g the model builder).
        import trimesh  # noqa: PLC0415

        dirs = self.dirs
        meshes = []

        # ------------------------------------------------------------------------------------------------------------
        # Load table 0
        object = trimesh.creation.box(extents=0.3 * dirs.right + 0.05 * dirs.up + 0.3 * dirs.front)
        params = {
            "hydroelastic_modulus": 1e3,
        }
        meshes.append(hydroelastic_loaders.generate_mesh(object.vertices, object.faces, params))
        meshes[-1].mu_static = wp.float32(1.0)
        meshes[-1].mu_dynamic = wp.float32(0.8)
        meshes[-1].mass = 1.0
        meshes[-1].compute_mesh_density = True
        meshes[-1].hunt_crossley_dissipation = wp.float32(1.0)

        # ------------------------------------------------------------------------------------------------------------
        # Load table 2
        Tf = wp.transform(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity())
        object = trimesh.creation.cylinder(radius=0.2, height=0.01)
        params = {
            "hydroelastic_modulus": 1e3,
            "is_visible": True,
            "Tf": Tf,
        }
        meshes.append(hydroelastic_loaders.generate_mesh(object.vertices, object.faces, params))
        # Realistic values
        meshes[-1].mu_static = wp.float32(1.0)
        meshes[-1].mu_dynamic = wp.float32(0.8)

        # ------------------------------------------------------------------------------------------------------------
        # Load Object
        quat = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), 0.25 * wp.pi)
        Tf = wp.transform(wp.vec3f(0.0, 0.0, 0.0), quat)
        object_path = "./newton/examples/assets/experimental-hydroelastic-assets/graspgen_handwheel.obj"
        object = trimesh.load(object_path)
        if not object.is_watertight:
            print("Wheel is not watertight.")
        params = {
            "hydroelastic_modulus": 1e3,
            "is_visible": True,
            "Tf": Tf,
        }
        meshes.append(hydroelastic_loaders.generate_hard_mesh(object.vertices, object.faces, params))
        meshes[-1].mu_static = wp.float32(1.0)
        meshes[-1].mu_dynamic = wp.float32(0.5)
        meshes[-1].mass = 0.200
        meshes[-1].compute_mesh_density = True

        # ------------------------------------------------------------------------------------------------------------
        # Load gripper base
        object = trimesh.creation.box(extents=0.1 * dirs.right + 0.02 * dirs.up + 0.02 * dirs.front)
        params = {
            "hydroelastic_modulus": 5e3,
        }
        meshes.append(hydroelastic_loaders.generate_mesh(object.vertices, object.faces, params))
        meshes[-1].mu_static = wp.float32(0.6)  # wp.float32(5.0)
        meshes[-1].mu_dynamic = wp.float32(0.575)  # wp.float32(4.9)
        meshes[-1].mass = 0.05
        meshes[-1].compute_mesh_density = True

        # ------------------------------------------------------------------------------------------------------------
        # Robotique 140 finger dims = (0.0078, 0.0655, 0.027)
        # Load robotiq finger 1
        Tf = wp.transform(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity())
        object = trimesh.creation.box(extents=0.0078 * dirs.right + 0.0655 * dirs.up + 0.027 * dirs.front)
        # object = object.subdivide_to_size(0.002)
        params = {
            "hydroelastic_modulus": 5e3,
            "is_visible": True,
            "Tf": Tf,
        }
        meshes.append(hydroelastic_loaders.generate_mesh(object.vertices, object.faces, params))
        meshes[-1].mu_static = wp.float32(0.8)
        meshes[-1].mu_dynamic = wp.float32(0.7)
        meshes[-1].mass = 0.025
        meshes[-1].compute_mesh_density = True

        # Load robotiq finger 2
        quat = wp.quat_identity()  # wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), 0.5 * np.pi)
        Tf = wp.transform(wp.vec3f(0.0, 0.0, 0.0), quat)
        params = {
            "hydroelastic_modulus": 5e3,
            "is_visible": True,
            "Tf": Tf,
        }
        meshes.append(hydroelastic_loaders.generate_mesh(object.vertices, object.faces, params))
        meshes[-1].mu_static = wp.float32(0.8)
        meshes[-1].mu_dynamic = wp.float32(0.7)
        meshes[-1].mass = 0.025
        meshes[-1].compute_mesh_density = True

        # ==============================================================================================================
        # Define initial poses.
        unit_q = wp.quat_identity()

        table_offset = 0
        table_com_height = 0.5 * 0.05
        object_com_height = 0.5 * 0.1318 - table_com_height  # 0.16
        # ========================================================================
        # TODO: Refactor variable names for clarity.
        finger_com_height = 0.5 * 0.025
        contact_length = object_com_height + finger_com_height

        object_y = table_offset + (table_com_height + object_com_height)
        finger_y = table_offset + (table_com_height + 2.0 * object_com_height + finger_com_height - contact_length)
        tool_offset = 0.15

        poses = np.zeros((6, 7), dtype=np.float32)
        # Table 0
        body_idx = 0
        table_0 = body_idx
        poses[body_idx, :3] = table_offset * dirs.up
        poses[body_idx, 3:] = unit_q
        # Table 1
        body_idx += 1
        table_1 = body_idx
        poses[body_idx, :3] = 0.5 * dirs.right
        poses[body_idx, 3:] = unit_q
        # Object
        body_idx += 1
        object = body_idx
        poses[body_idx, :3] = object_y * dirs.up
        poses[body_idx, 3:] = unit_q
        # Gripper base
        body_idx += 1
        gripper_base = body_idx
        poses[body_idx, :3] = (finger_y + tool_offset) * dirs.up - 0.07 * dirs.front
        poses[body_idx, 3:] = unit_q
        # Finger 0 (The actual pose of the finger is set when adding the joint for finger 0)
        body_idx += 1
        finger_0 = body_idx
        poses[body_idx, :3] = poses[gripper_base, :3] + (-0.025 * dirs.right - 0.1 * dirs.up)
        poses[body_idx, 3:] = unit_q
        # Finger 1 (The actual pose of the finger is set when adding the joint for finger 1)
        body_idx += 1
        finger_1 = body_idx
        poses[body_idx, :3] = poses[gripper_base, :3] + (+0.025 * dirs.right - 0.1 * dirs.up)
        poses[body_idx, 3:] = unit_q

        self.init_poses = poses

        # ==============================================================================================================
        # Setup scene
        scene = newton.ModelBuilder(up_axis=self.up_axis)

        # Add bodies and shapes to the model
        hydroelastic_loaders.add_bodies(scene, poses, meshes)

        # Add joints
        # Add table 0
        xform = wp.transform(wp.vec3f(poses[table_0, :3]), wp.quat_identity())
        scene.add_joint_fixed(parent=-1, child=table_0, parent_xform=xform)

        # Add table 1
        xform = wp.transform(wp.vec3f(poses[table_1, :3]), wp.quat_identity())
        scene.add_joint_fixed(parent=-1, child=table_1, parent_xform=xform)

        # Add wheel
        scene.add_joint_free(parent=-1, child=object)

        scene.add_joint_d6(
            parent=-1,
            child=gripper_base,
            parent_xform=wp.transform(wp.vec3f(poses[gripper_base, :3]), wp.quat_identity()),
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=(1.0, 0.0, 0.0),
                    target=0.0,
                    target_ke=100.0,
                    target_kd=25.0,
                    mode=newton.JointMode.TARGET_POSITION,
                    limit_lower=-2.0,
                    limit_upper=2.0,
                ),
                newton.ModelBuilder.JointDofConfig(
                    axis=(0.0, 1.0, 0.0),
                    target=0.0,
                    target_ke=100.0,
                    target_kd=25.0,
                    mode=newton.JointMode.TARGET_POSITION,
                    limit_lower=-2.0,
                    limit_upper=2.0,
                ),
                newton.ModelBuilder.JointDofConfig(
                    axis=(0.0, 0.0, 1.0),
                    target=0.0,
                    target_ke=100.0,
                    target_kd=25.0,
                    mode=newton.JointMode.TARGET_POSITION,
                    limit_lower=-2.0,
                    limit_upper=2.0,
                ),
            ],
        )

        finger_joint_config = newton.ModelBuilder.JointDofConfig(
            axis=dirs.right,
            target=0.0,
            target_ke=100.0,
            target_kd=25.0,
            mode=newton.JointMode.TARGET_POSITION,
            limit_lower=-2.0,
            limit_upper=2.0,
        )

        scene.add_joint_d6(
            parent=gripper_base,
            child=finger_0,
            parent_xform=wp.transform(wp.vec3(poses[finger_0, :3] - poses[gripper_base, :3]), wp.quat_identity()),
            linear_axes=[finger_joint_config],
        )

        scene.add_joint_d6(
            parent=gripper_base,
            child=finger_1,
            parent_xform=wp.transform(wp.vec3(poses[finger_1, :3] - poses[gripper_base, :3]), wp.quat_identity()),
            linear_axes=[finger_joint_config],
        )

        # finalize Scene
        model = scene.finalize()

        # ==============================================================================================================
        # Set hydroelastic meshes inside the model.
        model.hydro_mesh = meshes

        # ==============================================================================================================
        # Set collision pairs.
        model.collision_pairs = [[table_0, object], [table_1, object], [finger_0, object], [finger_1, object]]

        return model

    def capture(self):
        if wp.get_device().is_cuda and self.use_cuda_graph:
            # with wp.ScopedCapture(force_module_load=False) as capture:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            # wp.capture_debug_dot_print(capture.graph, "./out/graph.dot", verbose=True)
        else:
            self.graph = None

    def simulate(self):
        # self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.2)
        for _ in range(self.sim_substeps):
            ## Forces could also be cleared inside the compute_contact_forces function.
            ## So, that we can compare the forces generated by collide vs the ones generated by the hydroelastic contact.
            self.state_0.clear_forces()
            self.compute_contact_surfaces()
            self.assign_control(self.control)
            self.compute_contact_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Single simulation step - now without controller."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        self.compute_control()

        # Computing contact surface again before rendering.
        self.compute_contact_surfaces()

        self.append_to_data_history()

    def compute_contact_surfaces(self):
        with wp.ScopedTimer("Computation of contact surfaces", print=False):
            # Compute inverse transform of body_q
            wp.launch(
                hydroelastic_utils.compute_body_q_inv_mat,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q],
                outputs=[self.body_q_inv_mat],
            )

            # Compute hydroelastic contact isosurface.
            for i in range(self.contacts.num_isosurfaces):
                hydroelastic_isosurface.compute_contact_polygons(
                    self.state_0.body_q,
                    self.body_q_inv_mat,
                    self.model.hydro_mesh[self.contacts.isosurface[i].body_a],
                    self.model.hydro_mesh[self.contacts.isosurface[i].body_b],
                    self.contacts.isosurface[i],
                )

    def compute_contact_forces(self):
        if self.contacts.use_hydroelastic_inside_solver:
            return
        with wp.ScopedTimer("Computation of contact forces", print=False):
            # Integrate over isosurface to compute forces and torques.
            if self.twist_convention == 0:  # newton convention
                for i in range(self.contacts.num_isosurfaces):
                    hydroelastic_wrenches.compute_isosurface_wrenches(
                        self.contacts.isosurface[i],
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.model.hydro_mesh[self.contacts.isosurface[i].body_a],
                        self.model.hydro_mesh[self.contacts.isosurface[i].body_b],
                        self.twist_convention_wp,
                    )

                    hydroelastic_wrenches.launch_add_wrench_to_body_f(
                        body_a=self.contacts.isosurface[i].body_a,
                        body_b=self.contacts.isosurface[i].body_b,
                        force=self.contacts.isosurface[i].force,
                        torque_a=self.contacts.isosurface[i].torque_a_body,
                        torque_b=self.contacts.isosurface[i].torque_b_body,
                        twist_convention=self.twist_convention_wp,
                        body_f=self.state_0.body_f,
                    )
            elif self.twist_convention == 1:  # featherstone convention
                for i in range(self.contacts.num_isosurfaces):
                    hydroelastic_wrenches.compute_isosurface_wrenches(
                        self.contacts.isosurface[i],
                        self.state_0.body_q,
                        self.solver.body_v_s,
                        self.model.hydro_mesh[self.contacts.isosurface[i].body_a],
                        self.model.hydro_mesh[self.contacts.isosurface[i].body_b],
                        self.twist_convention_wp,
                    )

                    hydroelastic_wrenches.launch_add_wrench_to_body_f(
                        body_a=self.contacts.isosurface[i].body_a,
                        body_b=self.contacts.isosurface[i].body_b,
                        force=self.contacts.isosurface[i].force,
                        torque_a=self.contacts.isosurface[i].torque_a,
                        torque_b=self.contacts.isosurface[i].torque_b,
                        twist_convention=self.twist_convention_wp,
                        body_f=self.state_0.body_f,
                    )
            elif self.twist_convention == 2:  # mujoco convention
                for i in range(self.contacts.num_isosurfaces):
                    hydroelastic_wrenches.compute_isosurface_wrenches(
                        self.contacts.isosurface[i],
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.model.hydro_mesh[self.contacts.isosurface[i].body_a],
                        self.model.hydro_mesh[self.contacts.isosurface[i].body_b],
                        self.twist_convention_wp,
                    )

                    hydroelastic_wrenches.launch_add_wrench_to_body_f(
                        body_a=self.contacts.isosurface[i].body_a,
                        body_b=self.contacts.isosurface[i].body_b,
                        force=self.contacts.isosurface[i].force,
                        torque_a=self.contacts.isosurface[i].torque_a_body,
                        torque_b=self.contacts.isosurface[i].torque_b_body,
                        twist_convention=self.twist_convention_wp,
                        body_f=self.state_0.body_f,
                    )

    def assign_control(self, control):
        control.joint_target.assign(self.joint_target)
        # control.joint_f.assign(self.joint_f)

    def compute_control(self):
        joint_target = self.control.joint_target.numpy()
        sim_time = self.sim_time
        # ========================================================================
        # From https://blog.robotiq.com/hubfs/support-files/Product-sheet-Adaptive-Grippers-EN.pdf
        # For a Robotique 85 finger:
        #  - Min force............................. 20N
        #  - Max force............................ 235N
        #  - Friction grip payload......5Kg.=.aprox 50N
        #  - Closing speed.............. 20 to 150 mm/s
        # For a Robotique 140 finger: the max force is around 125N.
        #  - Min force............................. 10N
        #  - Max force............................ 125N
        #  - Friction grip payload....2.5Kg.=.aprox 25N
        #  - Closing speed.............. 30 to 250 mm/s
        # ========================================================================
        # From https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf
        # For a Panda finger:
        #  - Min force............................. 30N
        #  - Max force............................. 70N
        #  - Franka payload.............3KG.=.aprox 30N
        #  - Gripper weight.........0.73KG.=.aprox 7.2N
        #  - Closing speed max ................ 50 mm/s
        # ========================================================================
        # This has been rewritten to use  (front=x, right=y, up=z)

        t_down = 0.5
        t_close = t_down + 1.0
        t_lift = t_close + 2.0
        t_wait_after_lift = t_lift + 1.0
        t_shake_side = t_wait_after_lift + 1.0
        t_wait_after_shake = t_shake_side + 0.5
        t_move_to_table = t_wait_after_shake + 0.5
        t_go_down = t_move_to_table + 0.5
        t_open_fingers = t_go_down + 0.5
        t_wait_after_open_fingers = t_open_fingers + 0.5
        t_up = t_wait_after_open_fingers + 0.5

        target_down = 0.05
        target_close = 0.06
        lift_height = 0.15
        shake_side_distance = 0.1
        target_down_2 = 0.15
        target_up = 0.15

        joint_offset = 6
        freq = np.pi
        if sim_time < t_down:
            alpha = 0.5 * (1.0 - np.cos(freq * sim_time / t_down))
            joint_target[joint_offset + 2] = -target_down * alpha
        elif sim_time < t_close:
            joint_target[joint_offset + 2] = -target_down
            alpha = 0.5 * (1.0 - np.cos(freq * (sim_time - t_down) / (t_close - t_down)))
            joint_offset = 9
            joint_target[joint_offset + 0] = target_close * alpha
            joint_target[joint_offset + 1] = -joint_target[joint_offset + 0]
        elif sim_time < t_lift:
            t = (sim_time - t_close) / (t_lift - t_close)
            alpha = 1.0
            if t < 0.5:
                alpha = 0.5 * (1.0 - np.cos(2 * freq * t))
            joint_target[joint_offset + 2] = -target_down + lift_height * alpha
            joint_offset = 9
            joint_target[joint_offset + 0] = target_close
            joint_target[joint_offset + 1] = -joint_target[joint_offset + 0]
        elif sim_time < t_wait_after_lift:
            joint_target[joint_offset + 2] = -target_down + lift_height
        elif sim_time < t_shake_side:
            num_cycles = 4
            t = (sim_time - t_wait_after_lift) / (t_shake_side - t_wait_after_lift)
            alpha = np.sin(2 * num_cycles * freq * t)
            joint_target[joint_offset + 1] = shake_side_distance * alpha
        elif sim_time < t_wait_after_shake:
            joint_target[joint_offset + 1] = 0.0
        elif sim_time < t_move_to_table:
            t = (sim_time - t_wait_after_shake) / (t_move_to_table - t_wait_after_shake)
            alpha = 0.5 * (1.0 - np.cos(freq * t))
            joint_target[joint_offset + 1] = 0.5 * alpha
        elif sim_time < t_go_down:
            t = (sim_time - t_move_to_table) / (t_go_down - t_move_to_table)
            alpha = 0.5 * (1.0 - np.cos(freq * t))
            joint_target[joint_offset + 2] = -target_down + lift_height - target_down_2 * alpha
        elif sim_time < t_open_fingers:
            joint_target[joint_offset + 2] = -target_down + lift_height - target_down_2
            joint_offset = 9
            t = (sim_time - t_go_down) / (t_open_fingers - t_go_down)
            alpha = 0.5 * (1.0 - np.cos(freq * t))
            joint_target[joint_offset + 0] = target_close - 0.05 * alpha
            joint_target[joint_offset + 1] = -joint_target[joint_offset + 0]
        elif sim_time < t_wait_after_open_fingers:
            joint_offset = 9
            joint_target[joint_offset + 0] = 0.01
            joint_target[joint_offset + 1] = -joint_target[joint_offset + 0]
        elif sim_time < t_up:
            t = (sim_time - t_wait_after_open_fingers) / (t_up - t_wait_after_open_fingers)
            alpha = 0.5 * (1.0 - np.cos(freq * t))
            joint_target[joint_offset + 2] = -target_down + lift_height - target_down_2 + target_up * alpha

        self.joint_target.assign(wp.array(joint_target, dtype=wp.float32))

    def render(self):
        # Render the scene
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.render_forces()
        self.render_visuals()
        self.render_isosurface()
        self.viewer.end_frame()

        self.plot()
        self.frame += 1

        if isinstance(self.viewer, newton.viewer.ViewerUSD) and self.frame % 10 == 0:
            time_elapsed = time.perf_counter() - self.time_start
            print(
                f"Frame {self.frame}. Sim time: {self.sim_time:.2f}s. Walltime: {time_elapsed:.2f}s, Factor: {100.0 * self.sim_time / time_elapsed:.2f}%"
            )

    def append_to_data_history(self):
        num_isosurfaces = len(self.contacts.isosurface)
        num_bodies = self.model.body_count
        num_data = num_bodies + num_isosurfaces
        np_data = np.zeros((num_data, 9))
        for i in range(num_isosurfaces):
            torque = self.contacts.isosurface[i].torque_a_body.numpy()[0, :]
            force_n = self.contacts.isosurface[i].force_n.numpy()[0, :]
            force_t = self.contacts.isosurface[i].force_t.numpy()[0, :]
            np_data[i, 0:3] = torque
            np_data[i, 3:6] = force_n
            np_data[i, 6:9] = force_t

        body_q_np = self.state_0.body_q.numpy()
        body_qd_np = self.state_0.body_qd.numpy()
        if self.twist_convention == 1:
            body_qd_np = self.solver.body_v_s.numpy()
        for i in range(num_bodies):
            com = body_q_np[i, 0:3]
            omega = body_qd_np[i, 0:3]
            vs = body_qd_np[i, 3:6]
            p_com = vs
            if self.twist_convention == 1:
                p_com += wp.cross(omega, com)
            elif self.twist_convention == 2:
                quat = wp.quat(body_q_np[i, 3:7])
                Tf = wp.transform(wp.vec3f(0.0, 0.0, 0.0), quat)
                omega = np.array(wp.transform_vector(Tf, wp.vec3f(omega)))
                # omega = body_q_np[i, 3:7]
            np_data[num_isosurfaces + i, 0:3] = omega
            np_data[num_isosurfaces + i, 3:6] = p_com

        self.data_history.append(np_data)

    def plot_with_legend(self, t, data, ax, subplot_id):
        legend_title = f"min: {np.min(data):.2E}\nmax: {np.max(data):.2E}\nl: {data[-1]:.2E}"
        ax[subplot_id].plot(t, data)
        ax[subplot_id].legend(title=legend_title)

    def plot(self):
        if self.frame % 10 != 0:
            return

        if self.editable_vars.plot_flag and self.data_history and isinstance(self.viewer, newton.viewer.ViewerGL):
            import matplotlib.pyplot as plt  # noqa: PLC0415

            plt.ion()
            num_isosurfaces = len(self.contacts.isosurface)
            num_bodies = self.model.body_count

            # Plot the isosurfaces
            plot_id = 1
            if not plt.fignum_exists(plot_id):
                ncols = 3
                fig, axes = plt.subplots(num_isosurfaces, ncols, sharex=True, sharey=False)
                self.figs.append(fig)
                self.axs.append(axes)

            if plt.fignum_exists(plot_id):
                # fig.clear()
                t = np.arange(len(self.data_history)) * self.frame_dt
                for f in range(num_isosurfaces):
                    forces_n = np.zeros((4, len(self.data_history)))
                    forces_t = np.zeros((4, len(self.data_history)))
                    torques = np.zeros((4, len(self.data_history)))
                    for i in range(len(self.data_history)):
                        forces_n[0:3, i] = self.data_history[i][f, 3:6]
                        forces_n[3, i] = np.linalg.norm(forces_n[0:3, i])

                        forces_t[0:3, i] = self.data_history[i][f, 6:9]
                        forces_t[3, i] = np.linalg.norm(forces_t[0:3, i])

                        torques[0:3, i] = self.data_history[i][f, 0:3]
                        torques[3, i] = np.linalg.norm(torques[0:3, i])

                    axs_dims = len(self.axs[0].shape)
                    if axs_dims == 1:
                        if f == 0:
                            self.axs[0][3 * f + 0].set_title("Force_n")
                            self.axs[0][3 * f + 1].set_title("Force_t")
                            self.axs[0][3 * f + 2].set_title("Torque")

                        self.plot_with_legend(t, forces_n[3, :], self.axs[0], 3 * f + 0)
                        self.plot_with_legend(t, forces_t[3, :], self.axs[0], 3 * f + 1)
                        self.plot_with_legend(t, torques[3, :], self.axs[0], 3 * f + 2)

                    elif axs_dims == 2:
                        if f == 0:
                            self.axs[0][f][0].set_title("Force_n")
                            self.axs[0][f][1].set_title("Force_t")
                            self.axs[0][f][2].set_title("Torque")

                        self.plot_with_legend(t, forces_n[3, :], self.axs[0][f], 0)
                        self.plot_with_legend(t, forces_t[3, :], self.axs[0][f], 1)
                        self.plot_with_legend(t, torques[3, :], self.axs[0][f], 2)

                self.figs[plot_id - 1].canvas.draw()
                self.figs[plot_id - 1].canvas.flush_events()

            plot_id = 2
            index_offset = num_isosurfaces
            if not plt.fignum_exists(plot_id):
                ncols = 4
                fig, axes = plt.subplots(num_bodies, ncols, sharex=True, sharey=False)
                self.figs.append(fig)
                self.axs.append(axes)

            if plt.fignum_exists(plot_id):
                t = np.arange(len(self.data_history)) * self.frame_dt
                for v in range(num_bodies):
                    data = np.zeros((4, len(self.data_history)))
                    for i in range(len(self.data_history)):
                        data[0:3, i] = self.data_history[i][v + index_offset, 3:6]
                        data[3, i] = np.linalg.norm(data[0:3, i])

                    axs_dims = len(self.axs[plot_id - 1].shape)
                    if axs_dims == 1:
                        if v == 0:
                            self.axs[plot_id - 1][0].set_title("v_x")
                            self.axs[plot_id - 1][1].set_title("v_y")
                            self.axs[plot_id - 1][2].set_title("v_z")
                            self.axs[plot_id - 1][3].set_title("norm(v)")

                        self.plot_with_legend(t, data[3, :], self.axs[plot_id - 1], v)

                    elif axs_dims == 2:
                        if v == 0:
                            self.axs[plot_id - 1][0][0].set_title("v_x")
                            self.axs[plot_id - 1][0][1].set_title("v_y")
                            self.axs[plot_id - 1][0][2].set_title("v_z")
                            self.axs[plot_id - 1][0][3].set_title("norm(v)")

                        self.plot_with_legend(t, data[0, :], self.axs[plot_id - 1][v], 0)
                        self.plot_with_legend(t, data[1, :], self.axs[plot_id - 1][v], 1)
                        self.plot_with_legend(t, data[2, :], self.axs[plot_id - 1][v], 2)
                        self.plot_with_legend(t, data[3, :], self.axs[plot_id - 1][v], 3)

                self.figs[plot_id - 1].canvas.draw()
                self.figs[plot_id - 1].canvas.flush_events()

    def render_forces(self):
        hydroelastic_render_utils.render_forces(
            self.viewer,
            self.state_0,
            self.contacts,
            self.editable_vars,
        )

    def render_visuals(self):
        pass
        # hydroelastic_render_utils.render_visuals(self.viewer, self.state_0, self.visuals,)

    def render_isosurface(self):
        hydroelastic_render_utils.render_isosurfaces(
            self.viewer,
            self.state_0,
            self.contacts,
            self.editable_vars,
        )

    def setup_imgui(self):
        # Initialize ImGui manager
        if self.use_imgui:
            window_size = (300, 400)
            window_pos = (self.viewer.ui.io.display_size[0] - 310, 300)
            self.imgui_manager = hydroelastic_imgui.HydroelasticImGuiManager(self.viewer.ui, window_pos, window_size)
            self.imgui_manager.example = self
            self.viewer.register_ui_callback(lambda imgui: self.imgui_manager.draw_ui(imgui), position="free")


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, verbose=args.verbose)

    # Run example
    newton.examples.run(example)
