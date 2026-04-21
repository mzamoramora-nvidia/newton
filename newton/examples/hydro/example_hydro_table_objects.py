# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Hydro Table Objects
#
# Standalone contact benchmarking demo that drops 8 different object shapes
# onto a table to measure contact quality across collision pipelines.
# No robot, no IK, no gripper -- just objects settling under gravity.
#
# Each world gets one object shape (round-robin across 8 types). The state
# machine runs through DROP -> SETTLE -> DONE while tracking table force,
# penetration, and object velocity metrics.
#
# Command: python -m newton.examples hydro_table_objects
#
###########################################################################

from dataclasses import replace
from enum import IntEnum

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.solvers
import newton.viewer
from newton import Contacts
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact


class ObjectShape(IntEnum):
    BOX = 0
    SPHERE = 1
    CYLINDER = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CUP = 5
    RUBBER_DUCK = 6
    BUNNY = 7


SHAPE_NAMES = [s.name for s in ObjectShape]
NUM_SHAPES = len(ObjectShape)


class CollisionMode(IntEnum):
    MUJOCO = 0
    NEWTON_DEFAULT = 1
    NEWTON_SDF = 2
    NEWTON_HYDROELASTIC = 3


class Phase(IntEnum):
    DROP = 0
    SETTLE = 1
    DONE = 2


PHASE_NAMES = [p.name for p in Phase]
NUM_PHASES = len(Phase)


def _load_mesh_asset_no_sdf(asset_path, prim_path):
    """Load a USD mesh asset with scale baked in. SDF built later per-world."""
    stage = Usd.Stage.Open(str(asset_path / "model.usda"))
    prim = stage.GetPrimAtPath(prim_path)
    mesh = newton.usd.get_mesh(prim, load_normals=True, face_varying_normal_conversion="vertex_splitting")
    parent_prim = stage.GetPrimAtPath("/root/Model")
    scale = np.asarray(newton.usd.get_scale(parent_prim), dtype=np.float32)
    if not np.allclose(scale, 1.0):
        mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
    return mesh


def _load_bunny_mesh():
    """Load Stanford bunny from example assets. SDF built later per-world."""
    usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
    mesh_vertices = np.array(usd_geom.GetPointsAttr().Get(), dtype=np.float32)
    mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    # Bunny is Y-up; swap Y and Z so it stands upright in our Z-up world
    mesh_vertices = mesh_vertices[:, [0, 2, 1]]
    # Center at origin so bounding-box scaling works correctly
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0)) / 2
    mesh_vertices -= center
    return newton.Mesh(mesh_vertices, mesh_indices)


class Example:
    def __init__(self, viewer, args):
        self.test_mode = args.test
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.collide_substeps = args.collide_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = args.world_count
        self.seed = args.seed
        self.collision_mode = CollisionMode[args.collision_mode.upper()]
        self.kh = args.kh
        self.table_half_xy = args.table_half_xy
        self.hydro_table = args.hydro_table
        self.viewer = viewer
        self.episode_steps = 0

        # Phase transition times (seconds)
        self.drop_end = 0.5
        self.settle_end = 2.0

        self._generate_world_params()
        self._load_mesh_objects()
        scene = self._build_scene()
        self._setup_collision_sdf(scene)
        self.model = scene.finalize()

        # Contact budget
        self.rigid_contact_max = 2_000 * self.world_count
        print(
            f"Bodies: {self.model.body_count}, Joints: {self.model.joint_count}, DOFs: {self.model.joint_coord_count}"
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.control = self.model.control()

        self._setup_contact_sensor()
        self._create_collision_pipeline()
        self._create_solver()
        self._create_sensor_contacts()
        self._setup_metrics()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.0, 0.0, 0.8), -25, -140)
        self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True
        self._setup_gui()
        self.capture()

    def _generate_world_params(self):
        """Generate per-world object parameters: shape and size."""
        rng = np.random.default_rng(self.seed)
        wc = self.world_count

        # Round-robin shape assignment across 8 object types
        self.world_shapes = [ObjectShape(i % NUM_SHAPES) for i in range(wc)]

        # Fixed density
        self.object_density = 1000.0

        # Base half-size 25 mm with uniform variation factor
        base_hs = 0.025
        self.world_half_sizes = base_hs * rng.uniform(0.75, 1.25, size=wc)

    def _load_mesh_objects(self):
        """Load mesh assets for CUP, RUBBER_DUCK, BUNNY (only those actually needed)."""
        needed = set(self.world_shapes)
        self.mesh_objects = {}

        if ObjectShape.CUP in needed:
            cup_path = newton.utils.download_asset("manipulation_objects/cup")
            self.mesh_objects[ObjectShape.CUP] = _load_mesh_asset_no_sdf(cup_path, "/root/Model/Model")

        if ObjectShape.RUBBER_DUCK in needed:
            duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
            self.mesh_objects[ObjectShape.RUBBER_DUCK] = _load_mesh_asset_no_sdf(duck_path, "/root/Model/SurfaceMesh")

        if ObjectShape.BUNNY in needed:
            self.mesh_objects[ObjectShape.BUNNY] = _load_bunny_mesh()

    def _add_object(self, builder: newton.ModelBuilder, world_id: int):
        """Add an object to the builder for one world. All shapes as meshes."""
        shape = self.world_shapes[world_id]
        hs = float(self.world_half_sizes[world_id])

        # Object 0.5mm above the table surface
        z_gap = 0.0005
        table_surface_z = 0.1  # table center at z=0.05, half-extent 0.05
        obj_z = table_surface_z + hs + z_gap
        obj_xform = wp.transform(wp.vec3(0.0, 0.0, obj_z), wp.quat_identity())
        obj_body = builder.add_body(xform=obj_xform, label="object")

        # Create mesh for this shape
        if shape == ObjectShape.BOX:
            mesh = newton.Mesh.create_box(hs, hs, hs, compute_inertia=False)
        elif shape == ObjectShape.SPHERE:
            mesh = newton.Mesh.create_sphere(hs, compute_inertia=False)
        elif shape == ObjectShape.CYLINDER:
            mesh = newton.Mesh.create_cylinder(hs, hs, up_axis=newton.Axis.X, compute_inertia=False)
        elif shape == ObjectShape.CAPSULE:
            mesh = newton.Mesh.create_capsule(hs, hs, up_axis=newton.Axis.X, compute_inertia=False)
        elif shape == ObjectShape.ELLIPSOID:
            mesh = newton.Mesh.create_ellipsoid(hs * 2.0, hs, hs, compute_inertia=False)
        elif shape in (ObjectShape.CUP, ObjectShape.RUBBER_DUCK, ObjectShape.BUNNY):
            mesh = self.mesh_objects[shape]
        else:
            raise ValueError(f"Unknown object shape: {shape}")

        # Compute scale for mesh objects to fit target size
        if shape in (ObjectShape.CUP, ObjectShape.RUBBER_DUCK, ObjectShape.BUNNY):
            verts = mesh.vertices
            extents = verts.max(axis=0) - verts.min(axis=0)
            max_extent = float(extents.max())
            target_extent = 2.0 * hs
            sc = target_extent / max_extent if max_extent > 0 else 1.0
            scale = wp.vec3(sc, sc, sc)
        else:
            scale = wp.vec3(1.0, 1.0, 1.0)

        obj_cfg = replace(self.shape_cfg, density=self.object_density)
        builder.add_shape_mesh(
            body=obj_body,
            mesh=mesh,
            scale=scale,
            cfg=obj_cfg,
            label="object_shape",
        )
        return obj_body

    def _build_scene(self):
        """Build the multi-world scene with per-world table + object."""
        self.shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=self.kh,
            gap=0.0005,
            mu=1.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )

        # Table mesh: half-extents (table_half_xy, table_half_xy, 0.05) at center (0, 0, 0.05)
        table_mesh = newton.Mesh.create_box(self.table_half_xy, self.table_half_xy, 0.05, compute_inertia=False)
        table_cfg = replace(self.shape_cfg, density=0.0)

        scene = newton.ModelBuilder()
        scene.default_shape_cfg = self.shape_cfg
        scene.rigid_gap = self.shape_cfg.gap
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)

        for world_id in range(self.world_count):
            scene.begin_world()

            # Kinematic table body
            table_body = scene.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
                label="table",
                is_kinematic=True,
            )
            scene.add_shape_mesh(
                body=table_body,
                mesh=table_mesh,
                cfg=table_cfg,
                label="table_shape",
            )

            # Object above table
            obj_body = self._add_object(scene, world_id)

            if world_id == 0:
                self.object_body_offset = obj_body

            scene.end_world()

        scene.add_ground_plane()
        return scene

    def _setup_collision_sdf(self, builder):
        """Build SDFs on all collision shapes; mark object and table as hydroelastic."""
        if self.collision_mode not in (CollisionMode.NEWTON_SDF, CollisionMode.NEWTON_HYDROELASTIC):
            return

        sdf_narrow_band = (-0.0015, 0.0015)
        use_hydro = self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC

        # Pass 1: build SDF on every collision shape
        for shape_idx in range(builder.shape_count):
            if not (builder.shape_flags[shape_idx] & newton.ShapeFlags.COLLIDE_SHAPES):
                continue

            label = builder.shape_label[shape_idx] if shape_idx < len(builder.shape_label) else ""
            is_object = "object" in label
            sdf_max_res = 128 if is_object else 64
            sdf_margin = 0.0002 if is_object else self.shape_cfg.gap

            if builder.shape_type[shape_idx] == newton.GeoType.BOX:
                hx, hy, hz = builder.shape_scale[shape_idx]
                mesh = newton.Mesh.create_box(
                    hx,
                    hy,
                    hz,
                    duplicate_vertices=True,
                    compute_normals=False,
                    compute_uvs=False,
                    compute_inertia=True,
                )
                mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)
                builder.shape_type[shape_idx] = newton.GeoType.MESH
                builder.shape_source[shape_idx] = mesh
                builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)

            elif builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is None:
                    continue
                scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                if not np.allclose(scale, 1.0):
                    mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
                    builder.shape_source[shape_idx] = mesh
                    builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    if mesh.sdf is not None:
                        mesh.clear_sdf()
                    mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)
                elif mesh.sdf is None:
                    mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)

        # Pass 2: hydroelastic flags on object + table shapes
        if use_hydro:
            hydro_count = 0
            for shape_idx, label in enumerate(builder.shape_label):
                short = label.split("/")[-1] if label else ""
                is_object = "object" in short
                is_table = "table" in short

                mark_hydro = is_object or (self.hydro_table and is_table)
                if mark_hydro:
                    builder.shape_gap[shape_idx] = self.shape_cfg.gap
                    builder.shape_material_mu[shape_idx] = self.shape_cfg.mu
                    builder.shape_material_mu_torsional[shape_idx] = self.shape_cfg.mu_torsional
                    builder.shape_material_mu_rolling[shape_idx] = self.shape_cfg.mu_rolling
                    builder.shape_material_kh[shape_idx] = self.kh
                    builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
                    hydro_count += 1

            print(f"[SDF setup] Marked {hydro_count} shapes as HYDROELASTIC")

    def _setup_contact_sensor(self):
        """Create a SensorContact on object bodies with table counterpart shapes."""
        self.contact_sensor_table = SensorContact(
            self.model,
            sensing_obj_bodies="object",
            counterpart_shapes="table*",
        )

    def _create_collision_pipeline(self):
        """Create collision pipeline based on collision mode."""
        if self.collision_mode == CollisionMode.MUJOCO:
            self.collision_pipeline = None
            self.contacts = None
        elif self.collision_mode == CollisionMode.NEWTON_DEFAULT:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                broad_phase="nxn",
            )
            self.contacts = self.collision_pipeline.contacts()
        elif self.collision_mode == CollisionMode.NEWTON_SDF:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                broad_phase="explicit",
                reduce_contacts=True,
            )
            self.contacts = self.collision_pipeline.contacts()
        elif self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                broad_phase="explicit",
                reduce_contacts=True,
                sdf_hydroelastic_config=HydroelasticSDF.Config(
                    output_contact_surface=hasattr(self.viewer, "renderer"),
                    buffer_fraction=1.0,
                    buffer_mult_iso=2,
                    buffer_mult_contact=2,
                    anchor_contact=True,
                ),
            )
            self.contacts = self.collision_pipeline.contacts()
        else:
            raise ValueError(f"Unknown collision mode: {self.collision_mode}")

    def _create_solver(self):
        """Create the MuJoCo solver."""
        use_mujoco_contacts = self.collision_mode == CollisionMode.MUJOCO
        nconmax_per_world = self.rigid_contact_max // self.world_count
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=use_mujoco_contacts,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=50,
            ls_iterations=100,
            impratio=10.0,
            njmax=nconmax_per_world,
            nconmax=nconmax_per_world,
        )

    def _create_sensor_contacts(self):
        """Create or assign the Contacts object used by the contact sensor."""
        if self.collision_mode == CollisionMode.MUJOCO:
            self.contacts = Contacts(
                self.solver.get_max_contact_count(),
                0,
                requested_attributes=self.model.get_requested_contact_attributes(),
            )

    def _setup_metrics(self):
        """Initialize per-world metric tracking arrays."""
        wc = self.world_count
        num_active_phases = 2  # DROP and SETTLE

        # Per-phase metrics: (world_count, 2) for DROP and SETTLE
        self.phase_table_force_sum = np.zeros((wc, num_active_phases))
        self.phase_table_force_max = np.zeros((wc, num_active_phases))
        self.phase_pen_sum = np.zeros((wc, num_active_phases))
        self.phase_pen_max = np.zeros((wc, num_active_phases))
        self.phase_vel_sum = np.zeros((wc, num_active_phases))
        self.phase_count = np.zeros((wc, num_active_phases), dtype=np.int64)

        # Current-frame metrics
        self.cur_table_force = np.zeros(wc)
        self.cur_penetration = np.zeros(wc)
        self.cur_object_vel = np.zeros(wc)

        # Running max
        self.max_table_force = np.zeros(wc)
        self.max_penetration = np.zeros(wc)

        # NaN tracking
        self.world_nan_frame = np.full(wc, -1, dtype=np.int64)

        # Read initial object masses from model
        body_ws = self.model.body_world_start.numpy()
        body_mass_np = self.model.body_mass.numpy()
        self.object_masses = np.array(
            [float(body_mass_np[int(body_ws[w]) + self.object_body_offset]) for w in range(wc)]
        )

    def _get_current_phase(self) -> Phase:
        """Determine current phase from simulation time."""
        if self.sim_time < self.drop_end:
            return Phase.DROP
        elif self.sim_time < self.settle_end:
            return Phase.SETTLE
        else:
            return Phase.DONE

    def _setup_gui(self):
        """Register the side-panel GUI callback with the viewer."""
        self.selected_world = 0
        self.show_isosurface = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = self.show_isosurface
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._gui_impl, position="side")

    def _gui_impl(self, imgui):
        wc = self.world_count

        # World selector
        changed, val = imgui.slider_int("World", self.selected_world, 0, wc - 1)
        if changed:
            self.selected_world = val
        w = self.selected_world

        imgui.separator()

        # Object description
        shape_name = SHAPE_NAMES[self.world_shapes[w]]
        mass = float(self.object_masses[w])
        hs_mm = float(self.world_half_sizes[w]) * 1000.0
        imgui.text(f"Shape: {shape_name}")
        imgui.text(f"Mass:  {mass:.4f} kg")
        imgui.text(f"Size:  {hs_mm:.1f} mm (half-size)")

        imgui.separator()

        # Current state
        phase = self._get_current_phase()
        imgui.text(f"State: {PHASE_NAMES[phase]}")

        imgui.separator()

        # Table force
        expected_f = mass * 9.81
        imgui.text(f"Table F:    {self.cur_table_force[w]:.2f} N")
        imgui.text(f"Max Tbl F:  {self.max_table_force[w]:.2f} N")
        imgui.text(f"Expected F: {expected_f:.2f} N")

        imgui.separator()

        # Penetration
        pen = float(self.cur_penetration[w]) * 1000.0
        pen_max = float(self.max_penetration[w]) * 1000.0
        imgui.text(f"Penetration: {pen:.3f} mm")
        imgui.text(f"Max Pen:     {pen_max:.3f} mm")

        imgui.separator()

        # Object velocity
        vel = float(self.cur_object_vel[w]) * 1000.0
        imgui.text(f"Velocity: {vel:.2f} mm/s")

        # NaN indicator
        nan_frame = int(self.world_nan_frame[w])
        if nan_frame >= 0:
            imgui.text(f"NaN at frame {nan_frame}!")

        imgui.separator()
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC and hasattr(self.viewer, "renderer"):
            changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
            if changed:
                self.viewer.show_hydro_contact_surface = self.show_isosurface
        imgui.text(f"Frame: {self.episode_steps}  t={self.sim_time:.2f}s")

    def _update_metrics(self):
        """Update contact sensor readings and per-world metrics."""
        wc = self.world_count
        phase = self._get_current_phase()

        # Update contact forces via the solver and sensor
        self.solver.update_contacts(self.contacts, self.state_0)
        self.contact_sensor_table.update(self.state_0, self.contacts)

        # Read table forces from force_matrix
        table_fm = self.contact_sensor_table.force_matrix.numpy()

        for w in range(wc):
            tf = float(np.linalg.norm(table_fm[w].sum(axis=0)))
            self.cur_table_force[w] = tf
            self.max_table_force[w] = max(self.max_table_force[w], tf)

        # Read penetration from solver's MuJoCo data
        mj_data = self.solver.mjw_data
        nacon = min(int(mj_data.nacon.numpy()[0]), mj_data.naconmax)
        self.cur_penetration[:] = 0.0

        if nacon > 0:
            contact_dist = mj_data.contact.dist.numpy()
            contact_worldid = mj_data.contact.worldid.numpy()
            for c_idx in range(nacon):
                w = int(contact_worldid[c_idx])
                if 0 <= w < wc:
                    pen = max(0.0, -float(contact_dist[c_idx]))
                    self.cur_penetration[w] = max(self.cur_penetration[w], pen)

        for w in range(wc):
            self.max_penetration[w] = max(self.max_penetration[w], self.cur_penetration[w])

        # Track object velocity and NaN
        body_q_np = self.state_0.body_q.numpy()
        body_qd_np = self.state_0.body_qd.numpy()
        body_ws = self.model.body_world_start.numpy()

        for w in range(wc):
            obj_global = int(body_ws[w]) + self.object_body_offset
            obj_z = float(body_q_np[obj_global][2])

            # NaN detection
            if np.isnan(obj_z) and self.world_nan_frame[w] < 0:
                self.world_nan_frame[w] = self.episode_steps

            # Object velocity (linear part of spatial velocity)
            obj_vel = body_qd_np[obj_global][:3]
            vel_mag = float(np.linalg.norm(obj_vel))
            self.cur_object_vel[w] = vel_mag if not np.isnan(vel_mag) else 0.0

            # Bucket metrics by current phase
            if phase < Phase.DONE:
                p = int(phase)
                pen_mm = self.cur_penetration[w] * 1000
                self.phase_table_force_sum[w, p] += self.cur_table_force[w]
                self.phase_table_force_max[w, p] = max(self.phase_table_force_max[w, p], self.cur_table_force[w])
                self.phase_pen_sum[w, p] += pen_mm
                self.phase_pen_max[w, p] = max(self.phase_pen_max[w, p], pen_mm)
                if not np.isnan(vel_mag):
                    self.phase_vel_sum[w, p] += vel_mag * 1000  # mm/s
                self.phase_count[w, p] += 1

    def capture(self):
        self.graph_sim = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph_sim = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if self.collision_pipeline and i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph_sim is not None:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()
        self._update_metrics()
        self.sim_time += self.frame_dt
        self.episode_steps += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
            if (
                self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC
                and self.collision_pipeline is not None
                and self.collision_pipeline.hydroelastic_sdf is not None
            ):
                self.viewer.log_hydro_contact_surface(
                    self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
                    penetrating_only=True,
                )
        self.viewer.end_frame()

    def gui(self, imgui):
        self._gui_impl(imgui)

    def test_final(self):
        wc = self.world_count
        body_ws = self.model.body_world_start.numpy()
        body_mass_np = self.model.body_mass.numpy()

        results = []
        for w in range(wc):
            shape_name = SHAPE_NAMES[self.world_shapes[w]]
            obj_global = int(body_ws[w]) + self.object_body_offset
            mass = float(body_mass_np[obj_global])
            hs = float(self.world_half_sizes[w])
            has_nan = self.world_nan_frame[w] >= 0
            expected_force = mass * 9.81

            # Per-phase averages
            drop_p = int(Phase.DROP)
            settle_p = int(Phase.SETTLE)

            drop_cnt = max(1, int(self.phase_count[w, drop_p]))
            settle_cnt = max(1, int(self.phase_count[w, settle_p]))

            avg_tbl_f_drop = self.phase_table_force_sum[w, drop_p] / drop_cnt
            max_tbl_f_drop = self.phase_table_force_max[w, drop_p]
            avg_tbl_f_settle = self.phase_table_force_sum[w, settle_p] / settle_cnt
            max_tbl_f_settle = self.phase_table_force_max[w, settle_p]

            avg_pen_drop = self.phase_pen_sum[w, drop_p] / drop_cnt
            max_pen_drop = self.phase_pen_max[w, drop_p]
            avg_pen_settle = self.phase_pen_sum[w, settle_p] / settle_cnt
            max_pen_settle = self.phase_pen_max[w, settle_p]

            avg_vel_settle = self.phase_vel_sum[w, settle_p] / settle_cnt
            settled = avg_vel_settle < 1.0  # < 1 mm/s

            results.append(
                {
                    "world": w,
                    "shape": shape_name,
                    "mass": mass,
                    "half_size": hs,
                    "has_nan": has_nan,
                    "expected_force": expected_force,
                    "avg_tbl_f_drop": avg_tbl_f_drop,
                    "max_tbl_f_drop": max_tbl_f_drop,
                    "avg_tbl_f_settle": avg_tbl_f_settle,
                    "max_tbl_f_settle": max_tbl_f_settle,
                    "avg_pen_drop": avg_pen_drop,
                    "max_pen_drop": max_pen_drop,
                    "avg_pen_settle": avg_pen_settle,
                    "max_pen_settle": max_pen_settle,
                    "avg_vel_settle": avg_vel_settle,
                    "settled": settled,
                }
            )

        # --- Per-world results table ---
        w_col = 160
        print("\n" + "=" * w_col)
        print(f"  TABLE OBJECTS CONTACT BENCHMARK  (collision_mode={self.collision_mode.name})")
        print("=" * w_col)
        print(
            f"{'W':>3}  {'Shape':>12}  {'Mass':>6}  {'HS':>5}  "
            f"{'AvgTblF':>8}  {'MaxTblF':>8}  {'Expected':>8}  "
            f"{'AvgPen':>7}  {'MaxPen':>7}  {'AvgVel':>7}  {'Settled':>7}"
        )
        print("-" * w_col)
        for r in results:
            settled_str = "YES" if r["settled"] else "NO"
            print(
                f"{r['world']:3d}  {r['shape']:>12s}  {r['mass']:.3f}  "
                f"{r['half_size'] * 1000:5.1f}  "
                f"{r['avg_tbl_f_settle']:8.1f}  {r['max_tbl_f_settle']:8.1f}  "
                f"{r['expected_force']:8.2f}  "
                f"{r['avg_pen_settle']:7.2f}  {r['max_pen_settle']:7.2f}  "
                f"{r['avg_vel_settle']:7.1f}  {settled_str:>7s}"
            )

        # --- Per-shape aggregation ---
        print("\n" + "-" * 120)
        print("  PER-SHAPE AGGREGATION")
        print("-" * 120)
        print(
            f"{'Shape':>12}  {'Count':>5}  "
            f"{'AvgTblF':>8}  {'MaxTblF':>8}  {'Expected':>8}  "
            f"{'AvgPen':>7}  {'MaxPen':>7}  {'AvgVel':>7}  {'Settled':>7}"
        )
        print("-" * 120)
        for shape in ObjectShape:
            shape_results = [r for r in results if r["shape"] == shape.name]
            if not shape_results:
                continue
            n = len(shape_results)
            avg_tbl_f = np.mean([r["avg_tbl_f_settle"] for r in shape_results])
            max_tbl_f = np.max([r["max_tbl_f_settle"] for r in shape_results])
            avg_exp = np.mean([r["expected_force"] for r in shape_results])
            avg_pen = np.mean([r["avg_pen_settle"] for r in shape_results])
            max_pen = np.max([r["max_pen_settle"] for r in shape_results])
            avg_vel = np.mean([r["avg_vel_settle"] for r in shape_results])
            n_settled = sum(1 for r in shape_results if r["settled"])
            print(
                f"{shape.name:>12s}  {n:5d}  "
                f"{avg_tbl_f:8.1f}  {max_tbl_f:8.1f}  {avg_exp:8.2f}  "
                f"{avg_pen:7.2f}  {max_pen:7.2f}  {avg_vel:7.1f}  "
                f"{n_settled:3d}/{n:<3d}"
            )

        # --- Aggregate stats ---
        n_nan = sum(1 for r in results if r["has_nan"])
        n_settled = sum(1 for r in results if r["settled"])
        print("\n" + "-" * 80)
        print("  AGGREGATE STATISTICS")
        print("-" * 80)
        print(f"  Settled:            {n_settled}/{wc}")
        print(f"  NaN worlds:         {n_nan}/{wc}")
        print(f"  Penetration (mean): {np.mean(self.max_penetration) * 1000:.3f} mm")
        print(f"  Penetration (max):  {np.max(self.max_penetration) * 1000:.3f} mm")
        print(f"  Table force (mean): {np.mean(self.max_table_force):.1f} N")
        print(f"  Table force (max):  {np.max(self.max_table_force):.1f} N")
        avg_vel_all = np.mean([r["avg_vel_settle"] for r in results])
        print(f"  Avg velocity (settle): {avg_vel_all:.2f} mm/s")
        print("=" * w_col)

        # --- CI assertion: no NaN ---
        nan_ratio = n_nan / wc if wc > 0 else 0.0
        assert nan_ratio <= 0.25, (
            f"NaN detected in {n_nan}/{wc} world(s) ({nan_ratio:.0%}), exceeds 25% tolerance: "
            f"{[r['world'] for r in results if r['has_nan']]}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=200)
        parser.set_defaults(world_count=8)
        parser.add_argument(
            "--collision-mode",
            type=str,
            choices=["mujoco", "newton_default", "newton_sdf", "newton_hydroelastic"],
            default="newton_hydroelastic",
            help="Collision pipeline to use",
        )
        parser.add_argument("--kh", type=float, default=2e11, help="Hydroelastic stiffness [Pa]")
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per frame")
        parser.add_argument("--collide-substeps", type=int, default=4, help="Collide every N substeps")
        parser.add_argument("--seed", type=int, default=42, help="RNG seed for size variation")
        parser.add_argument(
            "--table-half-xy",
            type=float,
            default=0.4,
            help="Table half-extent in X and Y [m] (height fixed at 0.05m)",
        )
        parser.add_argument(
            "--hydro-table",
            action="store_true",
            help="Also mark table as HYDROELASTIC (default: SDF-only)",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
