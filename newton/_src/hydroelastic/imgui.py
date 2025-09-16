import time

import numpy as np


class HydroelasticImGuiManager:
    """An example ImGui manager that displays a few float values."""

    def __init__(self, ui, window_pos=(200, 10), window_size=(300, 200)):
        self.ui = ui
        # self.imgui = ui.imgui

        # UI properties
        self.window_pos = ui.imgui.ImVec2(window_pos[0], window_pos[1])
        self.window_size = ui.imgui.ImVec2(window_size[0], window_size[1])

        self.example = None
        self._first_frame = True

    def draw_ui(self, imgui):
        # set window position and size once
        imgui.set_next_window_pos(self.window_pos, imgui.Cond_.once)
        imgui.set_next_window_size(self.window_size, imgui.Cond_.once)

        imgui.begin("Hydroelastic UI")

        self.imgui_sim_info(imgui)
        self.imgui_camera_info(imgui)
        self.imgui_editable_vars(imgui)
        self.imgui_isosurfaces(imgui)
        self.imgui_state(imgui)
        # self.imgui_joint_target()
        imgui.end()

    def imgui_sim_info(self, imgui):
        if self._first_frame:
            self._start_time = time.time()
            self._last_time = time.perf_counter()
            self._first_frame = False

        time_ellapsed = time.time() - self._start_time
        imgui.text("time_ellapsed: " + f"{time_ellapsed:.1f}")

        if self.example is not None:
            if time_ellapsed > 0.0:
                time_factor = 100.0 * self.example.sim_time / time_ellapsed
                imgui.text("Approx. time_factor [%]: " + f"{time_factor:.2f}")

                now = time.perf_counter()
                dt = now - self._last_time
                self._last_time = now
                imgui.text("Approx. time_factor 2[%]: " + f"{(100.0 * self.example.frame_dt / dt):.2f}")

            imgui.text("frame: " + f"{self.example.frame}")
            imgui.text("frame_dt: " + f"{self.example.frame_dt:.2e}")
            imgui.text("sim_time: " + f"{self.example.sim_time:.2f}")
            imgui.text("sim_substeps: " + f"{self.example.sim_substeps}")
            imgui.text("sim_dt: " + f"{self.example.sim_dt:.2e}")
        imgui.separator()

    def imgui_camera_info(self, imgui):
        if self.example is None:
            return

        header_flags = 0
        cam_header_expanded = imgui.collapsing_header("Camera data", flags=header_flags)
        if cam_header_expanded:
            changed, values = imgui.input_float3("camera_pos", self.example.viewer.camera.pos)
            if changed:
                pos = type(self.example.viewer.camera.pos)(*values)
                self.example.viewer.camera.pos = pos

            changed, values = imgui.input_float("camera_pitch", self.example.viewer.camera.pitch)
            if changed:
                self.example.viewer.camera.pitch = values

            changed, values = imgui.input_float("camera_yaw", self.example.viewer.camera.yaw)
            if changed:
                self.example.viewer.camera.yaw = values

            imgui.separator()

    def imgui_editable_vars(self, imgui):
        if self.example is None:
            return

        if not hasattr(self.example, "editable_vars"):
            return

        if self.example.editable_vars is None:
            return

        editable_vars = self.example.editable_vars

        changed, values = imgui.input_float3("isosurface_offset", editable_vars.np_vertex_offset.tolist())
        if changed:
            editable_vars.np_vertex_offset = np.array(values, dtype=np.float32)

        # Only allow editing the sim_substeps if not using cuda graph
        if self.example.graph is None:
            changed, value = imgui.input_int("sim_substeps_editable", self.example.sim_substeps)
            if changed:
                self.example.sim_substeps = value
                self.example.sim_dt = self.example.frame_dt / self.example.sim_substeps

        changed, value = imgui.input_int("proxy_of_max_tet_pairs", editable_vars.proxy_of_max_tet_pairs)
        if changed:
            editable_vars.proxy_of_max_tet_pairs = value

        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("render_isosurfaces"):
            _, editable_vars.render_isosurfaces_edges = imgui.checkbox(
                "render_isosurfaces_edges", editable_vars.render_isosurfaces_edges
            )

            _, editable_vars.render_isosurfaces_normals = imgui.checkbox(
                "render_isosurfaces_normals", editable_vars.render_isosurfaces_normals
            )

            _, editable_vars.render_forces_flag = imgui.checkbox("render_forces_flag", editable_vars.render_forces_flag)

        imgui.separator()

        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("render_tet_meshes"):
            _, editable_vars.render_tet_mesh_edges = imgui.checkbox(
                "render_tet_mesh_edges", editable_vars.render_tet_mesh_edges
            )
        imgui.separator()

        _, editable_vars.plot_flag = imgui.checkbox("plot_flag", editable_vars.plot_flag)

        changed, value = imgui.input_float("force_scale", editable_vars.force_scale)
        if changed:
            editable_vars.force_scale = value

    def imgui_isosurfaces(self, imgui):
        # Isosurfaces info
        if self.example is None:
            return

        if not hasattr(self.example.contacts, "isosurface"):
            # print("example.contacts.isosurface not found")
            return

        if self.example.contacts.isosurface is None:
            return

        isosurfaces = self.example.contacts.isosurface

        header_flags = 0
        for i in range(len(isosurfaces)):
            body_a = isosurfaces[i].body_a
            body_b = isosurfaces[i].body_b
            isosurface_header_expanded = imgui.collapsing_header(f"Isosurfaces-{body_a}-{body_b}", flags=header_flags)
            if isosurface_header_expanded:
                imgui.text(f"max_tet_pairs_found: {isosurfaces[i].max_tet_pairs_found}")

                force = isosurfaces[i].force.numpy()[0, :]
                torque_a = isosurfaces[i].torque_a.numpy()[0, :]
                torque_b = isosurfaces[i].torque_b.numpy()[0, :]
                torque_a_body = isosurfaces[i].torque_a_body.numpy()[0, :]
                torque_b_body = isosurfaces[i].torque_b_body.numpy()[0, :]
                force_n = isosurfaces[i].force_n.numpy()[0, :]
                force_t = isosurfaces[i].force_t.numpy()[0, :]

                self.imgui_np_vec3(imgui, "force        ", force)
                self.imgui_np_vec3(imgui, "torque_a     ", torque_a)
                self.imgui_np_vec3(imgui, "torque_b     ", torque_b)
                self.imgui_np_vec3(imgui, "torque_a_body", torque_a_body)
                self.imgui_np_vec3(imgui, "torque_b_body", torque_b_body)
                self.imgui_np_vec3(imgui, "force_n      ", force_n)
                self.imgui_np_vec3(imgui, "force_t      ", force_t)

            imgui.separator()

    def imgui_np_vec3(self, imgui, label, vec3):
        imgui.text(f"{label} |{np.linalg.norm(vec3):.2e}|, [{vec3[0]:.2e}, {vec3[1]:.2e}, {vec3[2]:.2e}]")

    def imgui_state(self, imgui):
        if self.example is None:
            return

        if not hasattr(self.example, "state_0"):
            return

        if self.example.state_0 is None:
            return

        state = self.example.state_0

        header_flags = 0
        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()
        body_f = state.body_f.numpy()
        num_bodies = body_q.shape[0]
        for i in range(num_bodies):
            body_header_expanded = imgui.collapsing_header(f"Body-{i}", flags=header_flags)
            if body_header_expanded:
                imgui.text(f"pos[{i}]    : {body_q[i][0:3]}")
                imgui.text(f"quat[{i}]   : {body_q[i][3:7]}")
                imgui.text(f"ang_vel[{i}]: {body_qd[i][0:3]}")
                imgui.text(f"ang_vel_norm[{i}]: {np.linalg.norm(body_qd[i][0:3]):.2e}")
                imgui.text(f"vel[{i}]    : {body_qd[i][3:6]}")
                imgui.text(f"vel_norm[{i}]: {np.linalg.norm(body_qd[i][3:6]):.2e}")
                imgui.text(f"force[{i}]  : {body_f[i][3:6]}")
                imgui.text(f"force_norm[{i}]: {np.linalg.norm(body_f[i][3:6]):.2e}")
                imgui.text(f"torque[{i}] : {body_f[i][0:3]}")
                imgui.text(f"torque_norm[{i}]: {np.linalg.norm(body_f[i][0:3]):.2e}")

            imgui.separator()
        # imgui.text("state.body_q", self.state.body_q.numpy())
        # imgui.text("state.body_qd", self.state.body_qd.numpy())

    def imgui_joint_target(self, imgui):
        if self.example is None:
            return

        if not hasattr(self.example, "joint_target"):
            return

        if self.example.joint_target is None:
            return

        joint_target = self.example.joint_target

        imgui.text(f"joint_target: {joint_target.numpy()}")

        if self.init_poses is not None:
            for i in range(len(self.init_poses)):
                imgui.text(f"init_pose[{i}]: {self.init_poses[i]}")
