# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Heterogeneous Grasp
#
# Demonstrates heterogeneous grasping environments: each world contains a
# different object (shape, mass, size), all grasped and lifted by a Franka
# Panda + Robotiq 2F-85 gripper. Supports 4 collision pipelines: MuJoCo
# native contacts, Newton default, Newton SDF, and Newton hydroelastic.
#
# IMPORTANT: in MuJoCo collision mode every world must contain the same
# body topology, so every object is built as a mesh (primitives are
# converted via _PRIMITIVE_MESH_FACTORIES). The example uses
# separate_worlds=True; mixed primitive/mesh geom types across worlds
# would break that contract.
#
# A bare run shows a minimal info panel only. Tuning sliders, per-world
# metrics, debug-frame overlays, and success/lift statistics live on the
# attached GraspProbe in newton/tests/test_object_centric_grasp.py.
#
# Run commands (default mode is newton_hydroelastic):
#
#   # Interactive GL viewer with the default 24 worlds:
#   python -m newton.examples robot_heterogeneous_grasp
#
#   # Pick a collision mode:
#   python -m newton.examples robot_heterogeneous_grasp --collision-mode mujoco
#   python -m newton.examples robot_heterogeneous_grasp --collision-mode newton_default
#   python -m newton.examples robot_heterogeneous_grasp --collision-mode newton_sdf
#   python -m newton.examples robot_heterogeneous_grasp --collision-mode newton_hydroelastic
#
#   # Headless CI smoke (no probe; runs the NaN-guard fallback in test_final):
#   python -m newton.examples robot_heterogeneous_grasp --test --viewer null --quiet
#
# For the full tuning panel + debug-frame overlays + summary tables, attach
# a GraspProbe (see TestHeterogeneousGraspRegression in
# newton/tests/test_object_centric_grasp.py).
#
###########################################################################

import copy
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
import newton.viewer
from newton.geometry import HydroelasticSDF

_NUT_BOLT_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
_NUT_BOLT_ASSEMBLY = "m20_loose"

_LEGO_PITCH = 0.008
_LEGO_BRICK_HEIGHT = 0.0096
_LEGO_STUD_RADIUS = 0.0024
_LEGO_STUD_HEIGHT = 0.0017
_LEGO_WALL_THICKNESS = 0.0012
_LEGO_TOP_THICKNESS = 0.001
_LEGO_TUBE_OUTER_RADIUS = 0.003255
_LEGO_CYLINDER_SEGMENTS = 24

# Table dimensions. The table is centered at table_pos and extends along all
# three axes by these half-extents. _TABLE_HALF_X is large enough to support
# the Franka base at X = -0.5 plus the spawn region at X ~= 0; _TABLE_HALF_Y
# leaves a 5 cm margin around the spawn region. If you change either, also
# update the table SDF resolution in _setup_collision_sdf to keep the voxel
# size near 5 mm (the SDF builder aligns to 8-voxel chunks, so round to a
# multiple of 8).
_TABLE_HALF_X = 0.425
_TABLE_HALF_Y = 0.15
_TABLE_HEIGHT = 0.10

# Robotiq 2F-85 TCP offset: 174 mm from gripper base to fingertip center (closed).
_ROBOTIQ_TCP_OFFSET_M = 0.174
# Franka flange-to-Robotiq mount offset: 107 mm.
_PANDA_FLANGE_OFFSET_M = 0.107
# Hydroelastic stiffness for the table (Pa). Picked to dominate softer object materials
# so contact penetration is bounded by object compliance, not the table.
_TABLE_KH_PA = 1e10
# Vertical clearance between the spawn point and the object's lowest extent.
_GRASP_FLOOR_OFFSET_M = 0.0005


class ObjectShape(IntEnum):
    BOX = 0
    SPHERE = 1
    CYLINDER = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CUP = 5
    RUBBER_DUCK = 6
    LEGO_BRICK = 7
    RJ45_PLUG = 8
    BEAR = 9
    NUT = 10
    BOLT = 11


SHAPE_NAMES = [s.name for s in ObjectShape]
NUM_SHAPES = len(ObjectShape)

# Default object catalog: round-robin through every shape. Tests/benchmarks
# can narrow this by passing ``objects=[...]`` to ``Example.__init__``.
OBJECT_CATALOG_DEFAULT: list[ObjectShape] = list(ObjectShape)

# Primitives-only subset for fast smoke runs (no mesh-asset downloads).
OBJECT_CATALOG_PRIMITIVES: list[ObjectShape] = [
    ObjectShape.BOX,
    ObjectShape.SPHERE,
    ObjectShape.CYLINDER,
    ObjectShape.CAPSULE,
    ObjectShape.ELLIPSOID,
]

_MESH_SHAPES = frozenset(
    {
        ObjectShape.CUP,
        ObjectShape.RUBBER_DUCK,
        ObjectShape.LEGO_BRICK,
        ObjectShape.RJ45_PLUG,
        ObjectShape.BEAR,
        ObjectShape.NUT,
        ObjectShape.BOLT,
    }
)


class CollisionMode(IntEnum):
    MUJOCO = 0
    NEWTON_DEFAULT = 1
    NEWTON_SDF = 2
    NEWTON_HYDROELASTIC = 3


@dataclass(frozen=True)
class _CollisionModeSetup:
    """Per-mode dispatch table. Groups what used to be three separate cascades
    inside Example._create_solver / _create_collision_pipeline /
    _setup_collision_sdf into one record per collision mode.

    ``use_mujoco_contacts`` toggles the solver's internal contact detection.
    ``build_collision_pipeline`` returns the pipeline instance (or None for
    MuJoCo, which manages contacts internally).
    ``prepare_scene`` runs on the ModelBuilder before finalize and is None
    for modes that don't need SDF / hydroelastic prep.
    """

    use_mujoco_contacts: bool
    build_collision_pipeline: "Callable[[Example], newton.CollisionPipeline | None]"
    prepare_scene: "Callable[[Example, newton.ModelBuilder], None] | None" = None


def _build_pipeline_none(example) -> None:
    return None


def _build_pipeline_default(example):
    return newton.CollisionPipeline(
        example.model,
        rigid_contact_max=example.rigid_contact_max,
        broad_phase="nxn",
    )


def _build_pipeline_sdf(example):
    return newton.CollisionPipeline(
        example.model,
        rigid_contact_max=example.rigid_contact_max,
        broad_phase="explicit",
        reduce_contacts=True,
    )


def _build_pipeline_hydroelastic(example):
    return newton.CollisionPipeline(
        example.model,
        rigid_contact_max=example.rigid_contact_max,
        broad_phase="explicit",
        reduce_contacts=True,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            output_contact_surface=hasattr(example.viewer, "renderer"),
            buffer_fraction=1.0,
            buffer_mult_iso=2,
            buffer_mult_contact=2,
            anchor_contact=True,
        ),
    )


def _build_collision_sdfs(example, builder) -> None:
    """Pass 1 of SDF prep: build an SDF on every collision shape, converting
    boxes to meshes as needed. Shared by NEWTON_SDF and NEWTON_HYDROELASTIC.
    """
    sdf_narrow_band = (-0.0015, 0.0015)
    for shape_idx in range(builder.shape_count):
        if not (builder.shape_flags[shape_idx] & newton.ShapeFlags.COLLIDE_SHAPES):
            continue

        label = builder.shape_label[shape_idx] if shape_idx < len(builder.shape_label) else ""
        is_object = "object" in label
        is_table = "table" in label
        if is_object:
            sdf_max_res = 96
        elif is_table:
            # 0.85 m table / ~5 mm voxel target, rounded up to a multiple of 8 (tile-aligned).
            sdf_max_res = 176
        else:
            sdf_max_res = 64
        sdf_margin = 0.0002 if is_object else example.shape_cfg.gap

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


def _tag_hydroelastic_shapes(example, builder) -> None:
    """Pass 2 of SDF prep, hydroelastic-only: tag fingertip pads, object
    shapes, and the table with HYDROELASTIC so the contact-surface path has
    at least one pair to draw."""
    fingertip_names = {
        "left_pad1",
        "left_pad2",
        "right_pad1",
        "right_pad2",
        "left_follower_geom_1",
        "right_follower_geom_0",
    }

    hydro_count = 0
    for shape_idx, label in enumerate(builder.shape_label):
        short = label.split("/")[-1] if label else ""
        is_fingertip = short in fingertip_names
        is_object = "object" in short
        is_table = "table" in short

        if is_fingertip or is_object or is_table:
            cfg = example.shape_cfg
            builder.shape_gap[shape_idx] = cfg.gap
            builder.shape_material_mu[shape_idx] = cfg.mu
            builder.shape_material_mu_torsional[shape_idx] = cfg.mu_torsional
            builder.shape_material_mu_rolling[shape_idx] = cfg.mu_rolling
            builder.shape_material_kh[shape_idx] = _TABLE_KH_PA if is_table else example.kh
            builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
            if is_fingertip:
                builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.VISIBLE
            hydro_count += 1

    print(f"[SDF setup] Marked {hydro_count} shapes as HYDROELASTIC")


def _prepare_scene_sdf(example, builder) -> None:
    _build_collision_sdfs(example, builder)


def _prepare_scene_hydroelastic(example, builder) -> None:
    _build_collision_sdfs(example, builder)
    _tag_hydroelastic_shapes(example, builder)


_COLLISION_MODE_SETUPS: "dict[CollisionMode, _CollisionModeSetup]" = {
    CollisionMode.MUJOCO: _CollisionModeSetup(
        use_mujoco_contacts=True,
        build_collision_pipeline=_build_pipeline_none,
    ),
    CollisionMode.NEWTON_DEFAULT: _CollisionModeSetup(
        use_mujoco_contacts=False,
        build_collision_pipeline=_build_pipeline_default,
    ),
    CollisionMode.NEWTON_SDF: _CollisionModeSetup(
        use_mujoco_contacts=False,
        build_collision_pipeline=_build_pipeline_sdf,
        prepare_scene=_prepare_scene_sdf,
    ),
    CollisionMode.NEWTON_HYDROELASTIC: _CollisionModeSetup(
        use_mujoco_contacts=False,
        build_collision_pipeline=_build_pipeline_hydroelastic,
        prepare_scene=_prepare_scene_hydroelastic,
    ),
}


class TaskType(IntEnum):
    APPROACH = 0
    CLOSE_GRIPPER = 1
    SETTLE = 2  # hold pose + full closure; lets contact forces stabilize before LIFT
    LIFT = 3
    HOLD = 4
    DONE = 5


NUM_TASKS = len(TaskType)
TASK_NAMES = [t.name for t in TaskType]


# Per-shape vertical seed for the GUI's offset_local.z slider. Computed at init
# (see derive_offset_local_z) so the gripper starts above the table for every
# shape; the user retunes via the Apply button.
_GRASP_CLEARANCE = 0.05
_GRASP_Z_EXTRA = {
    ObjectShape.BOLT: 0.02,
    ObjectShape.BEAR: 0.01,
}


@dataclass(frozen=True)
class GraspSpec:
    """Per-shape grasp pose, authored in the object's COM frame.

    Both ``offset_local`` and ``quat_local`` are expressed in the body's COM-aligned
    frame. ``offset_local`` is in **units of the per-world object half-size** so the
    spec stays size-invariant under per-world half-size jitter; the kernel
    multiplies it by ``half_size`` to recover meters. ``quat_local`` rotates the EE
    target around the body, on top of the shared ``base_ee_rot``.

    Composed at runtime as:
        grasp_pos_world = com_world + body_q.q * (offset_local * half_size)
        grasp_rot_world = body_q.q * base_ee_rot * quat_local
        grasp_ctrl      = margin_pct_to_ctrl(margin_pct, y_half)
    """

    offset_local: wp.vec3 = field(default_factory=lambda: wp.vec3(0.0, 0.0, 0.0))
    quat_local: wp.quat = field(default_factory=wp.quat_identity)
    margin_pct: float = 0.05


GRASP_SPECS: dict[ObjectShape, GraspSpec] = {
    ObjectShape.BOX: GraspSpec(margin_pct=0.05),
    ObjectShape.SPHERE: GraspSpec(margin_pct=0.05),
    ObjectShape.CYLINDER: GraspSpec(margin_pct=0.05),
    ObjectShape.CAPSULE: GraspSpec(margin_pct=0.05),
    ObjectShape.ELLIPSOID: GraspSpec(margin_pct=0.05),
    ObjectShape.CUP: GraspSpec(margin_pct=0.22),
    ObjectShape.RUBBER_DUCK: GraspSpec(margin_pct=0.10),
    ObjectShape.LEGO_BRICK: GraspSpec(margin_pct=0.15),
    ObjectShape.RJ45_PLUG: GraspSpec(
        offset_local=wp.vec3(0.0, 0.30, 0.0),
        quat_local=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 90.0 * wp.pi / 180.0),
        margin_pct=0.10,
    ),
    ObjectShape.BEAR: GraspSpec(margin_pct=0.25),
    ObjectShape.NUT: GraspSpec(
        quat_local=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 30.0 * wp.pi / 180.0),
        margin_pct=0.15,
    ),
    ObjectShape.BOLT: GraspSpec(margin_pct=0.30),
}


class Example:
    def __init__(self, viewer, args, *, probe=None, objects: "list[ObjectShape] | None" = None):
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
        # Per-mode dispatch: pipeline factory + scene-preparation hook +
        # use_mujoco_contacts toggle. Looked up once at construction so the
        # three _create_* / _setup_* methods below stay 1-2 lines each.
        self._collision = _COLLISION_MODE_SETUPS[self.collision_mode]
        self.kh = args.kh
        self.verbose = args.verbose
        self.spawn_xy_range = args.spawn_xy_range
        self.spawn_yaw_range_rad = args.spawn_yaw_range * wp.pi / 180.0
        self.viewer = viewer
        self.episode_steps = 0
        # Object catalog to round-robin across worlds. ``None`` picks the full
        # 12-shape default catalog; tests/benchmarks can pass a narrower list
        # (e.g. ``OBJECT_CATALOG_PRIMITIVES``) to skip mesh-asset downloads.
        self.objects: list[ObjectShape] = list(objects) if objects is not None else list(OBJECT_CATALOG_DEFAULT)

        self._generate_world_params()
        robot_builder, arm_only_builder = self._build_robot()
        # IK model: arm-only (no gripper DOFs) for cleaner IK solving
        self.model_arm_only = arm_only_builder.finalize()
        self._load_mesh_objects()
        self._world_z_half = np.full(self.world_count, -1.0)
        self._world_y_half = np.full(self.world_count, -1.0)
        scene = self._build_scene(robot_builder)
        self._setup_collision_sdf(scene)
        self.model = scene.finalize()

        # Request the ``force`` contact attribute up-front so the collision
        # pipeline / MuJoCo Contacts buffer carries it regardless of whether
        # a GraspProbe (which adds SensorContact instances) is attached.
        self.model.request_contact_attributes("force")

        # Contact budget: 2000 per world (matching example_hydro_robotiq_gripper)
        self.rigid_contact_max = 2_000 * self.world_count
        print(
            f"Bodies: {self.model.body_count}, Joints: {self.model.joint_count}, DOFs: {self.model.joint_coord_count}"
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # IK and state machine setup
        self._setup_ik()
        self._setup_state_machine()
        self.control = self.model.control()
        self.joint_target_shape = self.control.joint_target_pos.reshape((self.world_count, -1)).shape
        self.joint_targets_2d = wp.zeros(self.joint_target_shape, dtype=wp.float32)
        self.graph_ik = None

        # MuJoCo ctrl for MJCF general actuators (CTRL_DIRECT mode)
        self.has_mujoco_ctrl = hasattr(self.control, "mujoco") and self.control.mujoco is not None
        if self.has_mujoco_ctrl:
            ctrl = self.control.mujoco.ctrl
            actuator_count = ctrl.shape[0] // self.world_count
            self.mujoco_ctrl_2d = ctrl.reshape((self.world_count, actuator_count))
            # Gripper actuator is the last one (after 7 arm actuators)
            self.gripper_actuator_idx = self.arm_dof_count
            print(f"MuJoCo ctrl: {actuator_count} actuators/world, gripper at idx {self.gripper_actuator_idx}")

            # Initialize ctrl to current arm joint positions so arm holds its pose
            init_q = self.model.joint_q.numpy()
            ctrl_np = ctrl.numpy().reshape(self.world_count, actuator_count)
            dofs_per_world = self.model.joint_coord_count // self.world_count
            for w in range(self.world_count):
                q_start = w * dofs_per_world
                for j in range(self.arm_dof_count):
                    ctrl_np[w, j] = init_q[q_start + j]
            wp.copy(ctrl, wp.array(ctrl_np.flatten(), dtype=wp.float32))

        self._create_collision_pipeline()
        self._create_solver()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.5, 0.5, 0.5), -15, -140)
        self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = False
        self.capture()

        # Optional diagnostic probe. Headless test runs (and the example
        # browser's reset path, which re-creates Example without kwargs) leave
        # this None and fall through to the minimal info GUI + NaN-guard
        # fallbacks installed below.
        self.probe = probe
        if probe is not None and hasattr(probe, "on_init"):
            probe.on_init(self)

    # ------------------------------------------------------------------
    # Public API: capture / simulate / step / render / test_final / create_parser
    # ------------------------------------------------------------------

    def capture(self):
        self.graph_sim = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph_sim = capture.graph
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if self.collision_pipeline and i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._set_joint_targets()
        if self.graph_sim is not None:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()
        if self.probe is not None and hasattr(self.probe, "on_step"):
            self.probe.on_step(self)
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
        if self.probe is not None and hasattr(self.probe, "on_render"):
            self.probe.on_render(self)
        self.viewer.end_frame()

    def gui(self, imgui):
        """Side-panel UI callback. Delegates to the probe when one is attached;
        otherwise draws a minimal info panel. The full tuning UI lives on the
        probe (see ``GraspProbe.on_gui_render`` in the tests).
        """
        if self.probe is not None and hasattr(self.probe, "on_gui_render"):
            self.probe.on_gui_render(self, imgui)
        else:
            self._draw_minimal_info_gui(imgui)

    def _draw_minimal_info_gui(self, imgui):
        """30-line info panel for bare runs with no probe attached."""
        imgui.text(f"Frame: {self.episode_steps}")
        imgui.text(f"Worlds: {self.world_count}")
        imgui.text(f"Collision mode: {self.collision_mode.name}")
        imgui.text(f"Substeps/frame: {self.sim_substeps}")
        imgui.text(f"Collide every: {self.collide_substeps} substep(s)")
        imgui.text(f"Sim time: {self.sim_time:.2f} s")
        imgui.separator()
        imgui.text("Attach a GraspProbe (see newton.tests.test_object_centric_grasp)")
        imgui.text("for tuning sliders, per-world metrics, and debug overlays.")

    def test_final(self):
        if self.probe is not None and hasattr(self.probe, "on_finish"):
            self.probe.on_finish(self)
            return
        body_q_np = self.state_0.body_q.numpy()
        nan_bodies = int(np.isnan(body_q_np).any(axis=-1).sum())
        assert nan_bodies == 0, f"NaN detected in {nan_bodies} body transform(s)"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=700)
        parser.set_defaults(world_count=24)
        parser.add_argument(
            "--collision-mode",
            type=str,
            choices=["mujoco", "newton_default", "newton_sdf", "newton_hydroelastic"],
            default="newton_hydroelastic",
            help="Collision pipeline to use",
        )
        parser.add_argument("--seed", type=int, default=42, help="RNG seed for mass/size variation")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument(
            "--spawn-xy-range",
            type=float,
            default=0.10,
            help="Half-range (m) of the uniform XY spawn-position randomization on the table. 0 disables.",
        )
        parser.add_argument(
            "--spawn-yaw-range",
            type=float,
            default=30.0,
            help="Half-range (deg) of the uniform Z-rotation spawn randomization. 0 disables.",
        )
        parser.add_argument("--kh", type=float, default=2e11, help="Hydroelastic stiffness [Pa]")
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per frame")
        parser.add_argument("--collide-substeps", type=int, default=4, help="Collide every N substeps")
        return parser

    # ------------------------------------------------------------------
    # Setup / build
    # ------------------------------------------------------------------

    def _download_assets(self):
        """Download menagerie assets once and cache paths."""
        if hasattr(self, "_franka_dir"):
            return
        self._franka_dir = newton.examples.download_external_git_folder(
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            "franka_emika_panda",
        )
        self._robotiq_dir = newton.examples.download_external_git_folder(
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            "robotiq_2f85_v4",
        )

    def _build_robot(self):
        """Build a ModelBuilder containing Franka Panda (no hand) + Robotiq 2F-85 V4."""
        self._download_assets()

        self.shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=self.kh,
            gap=0.0005,
            mu=1.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg = self.shape_cfg
        builder.rigid_gap = self.shape_cfg.gap
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # --- Franka Panda (no hand) from MuJoCo Menagerie ---
        # Scene centred on the world-Y axis: table at (0, 0), robot 0.5m behind it in -X.
        self.table_pos = wp.vec3(-0.275, 0.0, _TABLE_HEIGHT / 2.0)  # table body center, shifted under the robot
        self.spawn_center = wp.vec3(0.0, 0.0, _TABLE_HEIGHT)  # object spawn anchor on the table top
        self.robot_base_pos = wp.vec3(-0.5, 0.0, _TABLE_HEIGHT)  # on table, 0.5m behind center

        builder.add_mjcf(
            str(self._franka_dir / "panda_nohand.xml"),
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )

        # Initial arm pose (same as ik_cube_stacking — reach-forward config)
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[0:7] = init_q
        builder.joint_target_pos[0:7] = init_q

        # Save arm-only builder for IK (before attaching gripper)
        # The IK only needs the arm kinematic chain — link_offset handles TCP.
        arm_only_builder = copy.deepcopy(builder)

        def find_body(name):
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        # --- Robotiq 2F-85 V4 ---
        # Attach Robotiq to link7 (last link with mass), applying the attachment body's
        # transform plus an extra 90° Z rotation for better jaw alignment. The attachment
        # body in panda_nohand.xml is pos="0 0 0.107" quat="0.3826834 0 0 0.9238795";
        # MJCF stores quats (w, x, y, z) so we re-order to Warp's (x, y, z, w) below.
        q_attach = wp.quat(0.0, 0.0, 0.9238795, 0.3826834)
        q_90z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        builder.add_mjcf(
            str(self._robotiq_dir / "2f85.xml"),
            parent_body=find_body("link7"),
            xform=wp.transform(wp.vec3(0.0, 0.0, _PANDA_FLANGE_OFFSET_M), q_attach * q_90z),
            enable_self_collisions=False,
        )

        # Track key body indices.
        # The attachment body from panda_nohand.xml has zero mass/inertia;
        # give it a tiny mass so the inertia validator doesn't warn.
        self.ee_attachment_body_idx = find_body("attachment")
        builder.body_mass[self.ee_attachment_body_idx] = 1e-6
        builder.body_inertia[self.ee_attachment_body_idx] = wp.mat33(np.eye(3, dtype=np.float32) * 1e-9)
        franka_body_count = self.ee_attachment_body_idx + 1
        self.ee_base_body_idx = next(
            i for i in range(franka_body_count, builder.body_count) if builder.body_label[i].endswith("/base")
        )

        # --- Joint configuration ---
        # panda_nohand: 7 arm DOFs, no finger joints
        self.arm_dof_count = 7
        self.gripper_dof_start = self.arm_dof_count  # Robotiq DOFs start right after arm
        self.gripper_dof_count = builder.joint_coord_count - self.arm_dof_count

        print(
            f"Joint layout: {self.arm_dof_count} arm + "
            f"{self.gripper_dof_count} gripper = {builder.joint_coord_count} total DOFs"
        )
        print(f"Gripper DOF range: [{self.gripper_dof_start}, {self.gripper_dof_start + self.gripper_dof_count})")

        # Gripper armature scaling for hydroelastic stability (from example_hydro_robotiq_gripper)
        # The MJCF default armature is too small for hydroelastic contact stiffness.
        if self.gripper_dof_count == 6:
            mjcf_armature = np.array([0.005, 0.001, 0.001, 0.005, 0.001, 0.001])
        else:
            mjcf_armature = np.array([0.005, 0.001, 0.001, 0.001, 0.005, 0.001, 0.001, 0.001])
        armature_scale = 2.0
        gripper_armature = (armature_scale * mjcf_armature).tolist()
        builder.joint_armature[self.gripper_dof_start : self.gripper_dof_start + len(gripper_armature)] = (
            gripper_armature
        )

        # Gravity compensation on all bodies (arm + gripper)
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(self.arm_dof_count):
            gravcomp_attr.values[dof_idx] = True

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(1, builder.body_count):
            gravcomp_body.values[body_idx] = 1.0

        self.bodies_per_world = builder.body_count
        self.joints_per_world = builder.joint_count
        self.dofs_per_world = builder.joint_coord_count

        return builder, arm_only_builder

    def _generate_world_params(self):
        """Generate per-world object parameters: shape, mass, and size."""
        rng = np.random.default_rng(self.seed)
        n = self.world_count

        # Round-robin shape assignment across the injected object catalog.
        self.world_shapes = [self.objects[i % len(self.objects)] for i in range(n)]

        # Fixed density (1000 kg/m^3 ~= water): mass scales with shape volume and size,
        # so we don't have to combine an unrealistic density with the per-world size jitter.
        self.object_density = 1000.0

        # Per-world object half-size with +/- 25% uniform jitter around the base.
        base_half_size = 0.025
        self.world_half_sizes = base_half_size * rng.uniform(0.75, 1.25, size=n)

        # Per-world spawn pose randomization: XY offset on the table and Z-yaw rotation.
        # Both draws are uniform in [-range, +range]; ranges of 0 collapse to deterministic
        # spawns at the table center with the per-shape default orientation.
        self._world_spawn_xy = rng.uniform(-self.spawn_xy_range, self.spawn_xy_range, size=(n, 2)).astype(np.float32)
        self._world_spawn_yaw = rng.uniform(-self.spawn_yaw_range_rad, self.spawn_yaw_range_rad, size=n).astype(
            np.float32
        )

    def _load_mesh_objects(self):
        """Load mesh assets for all mesh-based shapes (only those actually needed).

        SDFs are NOT built here — they are built at the correct scale in
        ``_setup_collision_sdf`` after scale-baking per world.
        """
        needed = set(self.world_shapes)
        self.mesh_objects = {}

        if ObjectShape.CUP in needed:
            cup_path = newton.utils.download_asset("manipulation_objects/cup")
            self.mesh_objects[ObjectShape.CUP] = _load_mesh_asset_no_sdf(cup_path, "/root/Model/Model")

        if ObjectShape.RUBBER_DUCK in needed:
            duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
            self.mesh_objects[ObjectShape.RUBBER_DUCK] = _load_mesh_asset_no_sdf(duck_path, "/root/Model/SurfaceMesh")

        if ObjectShape.LEGO_BRICK in needed:
            self.mesh_objects[ObjectShape.LEGO_BRICK] = _lego_make_brick_mesh(4, 2)

        if ObjectShape.RJ45_PLUG in needed:
            rj45_path = newton.examples.get_asset("rj45_plug.usd")
            self.mesh_objects[ObjectShape.RJ45_PLUG] = _load_usd_mesh_local(rj45_path, "/World/Plug")

        if ObjectShape.BEAR in needed:
            bear_path = newton.examples.get_asset("bear.usd")
            self.mesh_objects[ObjectShape.BEAR] = _load_usd_mesh_local(bear_path, "/root/bear/bear")

        if ObjectShape.NUT in needed or ObjectShape.BOLT in needed:
            asset_path = newton.examples.download_external_git_folder(_NUT_BOLT_REPO_URL, _NUT_BOLT_FOLDER)
            if ObjectShape.NUT in needed:
                nut_file = str(asset_path / f"factory_nut_{_NUT_BOLT_ASSEMBLY}_subdiv_3x.obj")
                self.mesh_objects[ObjectShape.NUT] = _load_obj_mesh_trimesh(nut_file)
            if ObjectShape.BOLT in needed:
                bolt_file = str(asset_path / f"factory_bolt_{_NUT_BOLT_ASSEMBLY}.obj")
                self.mesh_objects[ObjectShape.BOLT] = _load_obj_mesh_trimesh(bolt_file)

    def _add_object(self, builder: newton.ModelBuilder, world_id: int):
        """Add a grasp object to the builder for one world."""
        shape = self.world_shapes[world_id]
        half_size = self.world_half_sizes[world_id]
        mesh = self.mesh_objects[shape] if shape in _MESH_SHAPES else _PRIMITIVE_MESH_FACTORIES[shape](half_size)

        if shape in _MESH_SHAPES:
            extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
            uniform_scale = 2.0 * half_size / extents.max() if extents.max() > 0 else 1.0
            # RJ45_PLUG is spawned with a 90° Z-yaw so its body-frame Y is the mesh's X-extent.
            y_extent = extents[0] if shape == ObjectShape.RJ45_PLUG else extents[1]
            self._world_y_half[world_id] = y_extent / 2.0 * uniform_scale
            self._world_z_half[world_id] = extents[2] / 2.0 * uniform_scale
        else:
            uniform_scale = 1.0
            self._world_y_half[world_id] = half_size
            self._world_z_half[world_id] = half_size
        scale = wp.vec3(uniform_scale, uniform_scale, uniform_scale)

        z_axis = wp.vec3(0.0, 0.0, 1.0)
        obj_rot = wp.quat_from_axis_angle(z_axis, wp.pi / 2.0) if shape == ObjectShape.RJ45_PLUG else wp.quat_identity()
        # Per-world spawn randomization: XY offset + Z-yaw composed onto obj_rot.
        sx, sy, sz = self.spawn_center
        dx, dy = self._world_spawn_xy[world_id]
        yaw_rot = wp.quat_from_axis_angle(z_axis, self._world_spawn_yaw[world_id])
        obj_z = sz + self._world_z_half[world_id] + _GRASP_FLOOR_OFFSET_M
        obj_body = builder.add_body(
            xform=wp.transform(wp.vec3(sx + dx, sy + dy, obj_z), yaw_rot * obj_rot),
            label="object",
        )

        obj_cfg = replace(self.shape_cfg, density=self.object_density)
        builder.add_shape_mesh(
            body=obj_body,
            mesh=mesh,
            scale=scale,
            cfg=obj_cfg,
            label=f"object_shape_{shape.name.lower()}",
        )

        # The LEGO brick gets explicit floor + four wall colliders inside its hollow shell;
        # other shapes get five tiny placeholder boxes so every world has identical body
        # topology (required by MuJoCo's separate_worlds=True mode).
        collider_cfg = replace(self.shape_cfg, density=0.0, is_visible=False)
        if shape == ObjectShape.LEGO_BRICK:
            sf = uniform_scale
            inset = 0.0001 * sf
            ox = _LEGO_PITCH * 2 * sf
            oy = _LEGO_PITCH * sf
            center_z = (_LEGO_BRICK_HEIGHT + _LEGO_STUD_HEIGHT) / 2.0 * sf
            box_hz = 0.5 * _LEGO_BRICK_HEIGHT * sf - inset
            box_cz = 0.5 * _LEGO_BRICK_HEIGHT * sf - center_z
            wt = _LEGO_WALL_THICKNESS * sf
            wall_hx = 0.5 * wt - inset
            wall_hy = 0.5 * wt - inset
            specs = [
                ((0.0, 0.0, box_cz), ox - inset, oy - inset, box_hz),
                ((ox - 0.5 * wt, 0.0, box_cz), wall_hx, oy - inset, box_hz),
                ((-(ox - 0.5 * wt), 0.0, box_cz), wall_hx, oy - inset, box_hz),
                ((0.0, oy - 0.5 * wt, box_cz), ox - inset, wall_hy, box_hz),
                ((0.0, -(oy - 0.5 * wt), box_cz), ox - inset, wall_hy, box_hz),
            ]
        else:
            t = 1.0e-4
            specs = [((0.0, 0.0, 0.0), t, t, t)] * 5
        for pos, hx, hy, hz in specs:
            builder.add_shape_box(
                body=obj_body,
                hx=hx,
                hy=hy,
                hz=hz,
                xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
                cfg=collider_cfg,
                label="object_collider",
            )

        return obj_body

    def _build_scene(self, robot_builder):
        """Build the multi-world scene with per-world robot + table + object."""
        # Table as kinematic body (mass=0) so MuJoCo detects collisions per world.
        # Using body=-1 (static) causes MuJoCo to miss collisions for replicated worlds.
        table_mesh = newton.Mesh.create_box(_TABLE_HALF_X, _TABLE_HALF_Y, _TABLE_HEIGHT / 2.0, compute_inertia=False)
        table_cfg = replace(self.shape_cfg, density=0.0)

        scene = newton.ModelBuilder()
        for world_id in range(self.world_count):
            scene.begin_world()
            scene.add_builder(robot_builder)

            # Kinematic table body (does not respond to forces)
            table_body = scene.add_body(
                xform=wp.transform(self.table_pos, wp.quat_identity()),
                label="table",
                is_kinematic=True,
            )
            scene.add_shape_mesh(
                body=table_body,
                mesh=table_mesh,
                cfg=table_cfg,
                color=(0.20, 0.20, 0.22),
                label="table_shape",
            )

            # Object on table surface at (0, 0, spawn_center.z + hs) plus the per-world spawn jitter
            obj_body = self._add_object(scene, world_id)

            # After first world: record the object's local body index
            if world_id == 0:
                self.object_body_offset = obj_body

            scene.end_world()

        scene.add_ground_plane()
        return scene

    def _setup_ik(self):
        """Create the IK solver with per-world problems.

        Uses the arm-only model (no gripper DOFs) for clean IK solving.
        The EE link is the ``attachment`` body at the Franka flange.
        The TCP offset (174mm) maps the IK target to the fingertip center.
        """
        # Find the attachment body in the arm-only model (last body)
        self.ik_ee_index = self.model_arm_only.body_count - 1

        # Evaluate FK to get initial EE pose
        state_single = self.model_arm_only.state()
        newton.eval_fk(self.model_arm_only, self.model_arm_only.joint_q, self.model_arm_only.joint_qd, state_single)
        body_q_np = state_single.body_q.numpy()
        ee_tf = wp.transform(*body_q_np[self.ik_ee_index])
        init_ee_pos = wp.transform_get_translation(ee_tf)
        init_ee_rot = wp.transform_get_rotation(ee_tf)

        # link_offset maps IK targets to the TCP, so grasp_pos values land at the
        # fingertip center rather than the gripper base. init target must be the
        # TCP position (not EE body) so the initial config is already a perfect
        # solution and the IK doesn't drift.
        tcp_offset_vec = wp.vec3(0.0, 0.0, _ROBOTIQ_TCP_OFFSET_M)
        init_tcp_pos = init_ee_pos + wp.quat_rotate(init_ee_rot, tcp_offset_vec)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_index,
            link_offset=tcp_offset_vec,
            target_positions=wp.array([init_tcp_pos] * self.world_count, dtype=wp.vec3),
        )

        # Rotation target: use the arm-only model's attachment body rotation.
        # The Robotiq is rigidly mounted with an extra 90° Z rotation, but
        # the IK only controls the arm — it should keep the attachment body
        # at its initial orientation.
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ik_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([list(init_ee_rot)] * self.world_count, dtype=wp.vec4),
        )

        # Joint limit objective — replicate single-model limits for all problems
        ik_dofs = self.model_arm_only.joint_coord_count
        ll = self.model_arm_only.joint_limit_lower.numpy()
        lu = self.model_arm_only.joint_limit_upper.numpy()
        joint_limit_lower = wp.array(np.tile(ll, self.world_count), dtype=wp.float32)
        joint_limit_upper = wp.array(np.tile(lu, self.world_count), dtype=wp.float32)
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=joint_limit_lower.flatten(),
            joint_limit_upper=joint_limit_upper.flatten(),
        )

        # IK solution array: (world_count, joint_coord_count)
        # Initialize from the multi-world joint_q, taking each world's slice
        joint_q_np = self.model.joint_q.numpy().reshape(self.world_count, -1)[:, :ik_dofs]
        self.joint_q_ik = wp.array(joint_q_np, dtype=wp.float32)

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_arm_only,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _setup_state_machine(self):
        """Initialize per-world state machine arrays and grasp parameters."""
        # Task state: index and timer per world
        self.task_idx = wp.zeros(self.world_count, dtype=wp.int32)
        self.task_timer = wp.zeros(self.world_count, dtype=wp.float32)

        # Task durations: [APPROACH, CLOSE_GRIPPER, SETTLE, LIFT, HOLD, DONE]
        durations = [1.5, 1.0, 0.5, 1.0, 1.5, 0.0]
        self.task_durations = wp.array(durations, dtype=wp.float32)

        # EE rotation target: use the arm-only model's attachment body rotation.
        # This matches the IK rotation objective so there's no mismatch.
        state_arm = self.model_arm_only.state()
        newton.eval_fk(self.model_arm_only, self.model_arm_only.joint_q, self.model_arm_only.joint_qd, state_arm)
        arm_ee_rot = wp.transform_get_rotation(wp.transform(*state_arm.body_q.numpy()[self.ik_ee_index]))

        # --- Object-centric grasp: per-world spec inputs + SoA runtime outputs ---
        self._shape_mask: dict[ObjectShape, np.ndarray] = {
            shape: np.where(np.asarray(self.world_shapes) == shape)[0].astype(np.int32)
            for shape in set(self.world_shapes)
        }

        # Per-world spec materialized as parallel CPU/GPU arrays. Fill the CPU side
        # from GRASP_SPECS, then construct the GPU mirror once.
        offset_np = np.zeros((self.world_count, 3), dtype=np.float32)
        quat_np = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (self.world_count, 1))
        ctrl_np = np.zeros(self.world_count, dtype=np.float32)
        for i, shape in enumerate(self.world_shapes):
            spec = GRASP_SPECS[shape]
            z_local = derive_offset_local_z(shape, half_size=self.world_half_sizes[i], z_half=self._world_z_half[i])
            offset_np[i] = (*spec.offset_local[:2], z_local)
            quat_np[i] = spec.quat_local
            ctrl_np[i] = margin_pct_to_ctrl(spec.margin_pct, y_half_m=self._world_y_half[i])
        self.spec = _PerWorldGraspSpec(
            offset_local=wp.array(offset_np, dtype=wp.vec3),
            quat_local=wp.array(quat_np, dtype=wp.quat),
            ctrl=wp.array(ctrl_np, dtype=wp.float32),
            offset_local_np=offset_np,
            quat_local_np=quat_np,
            ctrl_np=ctrl_np,
            offset_src=wp.zeros(1, dtype=wp.vec3),
            quat_src=wp.zeros(1, dtype=wp.quat),
            ctrl_src=wp.zeros(1, dtype=wp.float32),
        )

        # GPU runtime SoA outputs of compute_grasp_targets.
        self.grasp_pos = wp.zeros(self.world_count, dtype=wp.vec3)
        self.grasp_rot = wp.zeros(self.world_count, dtype=wp.quat)
        self.grasp_ctrl = wp.zeros(self.world_count, dtype=wp.float32)

        # Per-world half-size array for the init kernel
        self._world_half_size_array = wp.array(np.asarray(self.world_half_sizes, dtype=np.float32), dtype=wp.float32)

        # base_ee_rot as a wp.quat scalar (used by the init kernel)
        self.base_ee_rot = wp.quat(*arm_ee_rot)

        self.lift_distance_m: float = 0.1  # tunable via the GUI lift slider

        # state_0 is fully initialised by the time _setup_state_machine runs.
        wp.launch(
            compute_grasp_targets,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.model.body_com,
                self.model.body_world_start,
                self.object_body_offset,
                self._world_half_size_array,
                self.spec.offset_local,
                self.spec.quat_local,
                self.spec.ctrl,
                self.base_ee_rot,
            ],
            outputs=[self.grasp_pos, self.grasp_rot, self.grasp_ctrl],
        )

        # Target arrays (raw and interpolated)
        self.ee_pos_target = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(self.world_count, dtype=wp.vec4)
        self.ee_pos_target_interp = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target_interp = wp.zeros(self.world_count, dtype=wp.vec4)
        self.gripper_target = wp.zeros(self.world_count, dtype=wp.float32)

        # Actual EE position (extracted from body_q each frame)
        self.ee_pos_actual = wp.zeros(self.world_count, dtype=wp.vec3)

        # Global body indices for EE in each world. Using the attachment body (not the
        # Robotiq base) keeps EE tracking aligned with the IK model's EE link.
        body_ws = self.model.body_world_start.numpy()[: self.world_count].astype(np.int32)
        self.ee_body_global_indices = wp.array(body_ws + self.ee_attachment_body_idx, dtype=wp.int32)

        # Snapshot of body_q at the start of each task (for interpolation)
        # Initialize with current state_0 body_q
        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.body_count_total = self.model.body_count

    def _setup_collision_sdf(self, builder):
        """Run the mode's scene-preparation hook (build SDFs, tag hydroelastic
        shapes, etc.). No-op for modes without a hook."""
        if self._collision.prepare_scene is not None:
            self._collision.prepare_scene(self, builder)

    def _create_collision_pipeline(self):
        """Instantiate the collision pipeline via the mode's factory and adopt
        its Contacts buffer (or None for MuJoCo, which manages contacts internally)."""
        self.collision_pipeline = self._collision.build_collision_pipeline(self)
        self.contacts = self.collision_pipeline.contacts() if self.collision_pipeline is not None else None

    def _create_solver(self):
        """Create the MuJoCo solver. The only per-mode knob is whether the
        solver does its own contact detection (use_mujoco_contacts)."""
        nconmax_per_world = self.rigid_contact_max // self.world_count
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=self._collision.use_mujoco_contacts,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=100,
            ls_iterations=200,
            impratio=50.0,
            njmax=nconmax_per_world,
            nconmax=nconmax_per_world,
        )

    # ------------------------------------------------------------------
    # Per-step / per-frame internals
    # ------------------------------------------------------------------

    def apply_grasp_edits(
        self,
        w: int,
        *,
        offset_local: wp.vec3,
        quat_local: wp.quat,
        margin_pct: float,
        lift_distance_m: float,
        broadcast: bool,
    ) -> None:
        """Commit a grasp-spec edit for world w's shape to GRASP_SPECS and
        the GPU. Edits flow in via GraspProbe.on_gui_render."""
        shape = self.world_shapes[w]
        new_spec = GraspSpec(offset_local=offset_local, quat_local=quat_local, margin_pct=margin_pct)
        GRASP_SPECS[shape] = new_spec
        self.lift_distance_m = lift_distance_m

        affected = self._shape_mask[shape] if broadcast else np.array([w], dtype=np.int32)
        for idx in affected:
            idx_int = int(idx)
            self.spec.offset_local_np[idx_int] = new_spec.offset_local
            self.spec.quat_local_np[idx_int] = new_spec.quat_local
            self.spec.ctrl_np[idx_int] = margin_pct_to_ctrl(new_spec.margin_pct, y_half_m=self._world_y_half[idx_int])

        self.upload_grasp_targets(slot=None if broadcast else w)

    def upload_grasp_targets(self, slot: int | None) -> None:
        """Push spec buffers (CPU mirror -> GPU) and recompute grasp targets.

        ``slot=None`` rebuilds every world via a full-array assign + dim=world_count launch.
        ``slot=i`` patches just world ``i`` with a 1-element copy + dim=1 launch.
        """
        s = self.spec
        common_inputs = [
            self.state_0.body_q,
            self.model.body_com,
            self.model.body_world_start,
            self.object_body_offset,
            self._world_half_size_array,
            s.offset_local,
            s.quat_local,
            s.ctrl,
            self.base_ee_rot,
        ]
        outputs = [self.grasp_pos, self.grasp_rot, self.grasp_ctrl]
        if slot is None:
            s.offset_local.assign(s.offset_local_np)
            s.quat_local.assign(s.quat_local_np)
            s.ctrl.assign(s.ctrl_np)
            wp.launch(compute_grasp_targets, dim=self.world_count, inputs=common_inputs, outputs=outputs)
        else:
            s.offset_src.assign(s.offset_local_np[slot : slot + 1])
            s.quat_src.assign(s.quat_local_np[slot : slot + 1])
            s.ctrl_src.assign(s.ctrl_np[slot : slot + 1])
            wp.copy(s.offset_local, s.offset_src, dest_offset=slot, count=1)
            wp.copy(s.quat_local, s.quat_src, dest_offset=slot, count=1)
            wp.copy(s.ctrl, s.ctrl_src, dest_offset=slot, count=1)
            wp.launch(compute_grasp_targets_slot, dim=1, inputs=[slot, *common_inputs], outputs=outputs)

    def _set_joint_targets(self):
        """Compute IK targets from state machine and solve IK each frame."""
        # 1. Compute targets from state machine (with interpolation)
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_idx,
                self.task_timer,
                self.task_durations,
                self.grasp_pos,
                self.grasp_rot,
                self.grasp_ctrl,
                self.lift_distance_m,
                self.task_init_body_q,
                self.ee_body_global_indices,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_rot_target,
                self.ee_pos_target_interp,
                self.ee_rot_target_interp,
                self.gripper_target,
            ],
        )

        # 2. Update IK objectives with INTERPOLATED targets (smooth motion)
        self.pos_obj.set_target_positions(self.ee_pos_target_interp)
        self.rot_obj.set_target_rotations(self.ee_rot_target_interp)

        # 3. Solve IK
        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # 4. Write IK arm solution + gripper target to joint_targets
        wp.launch(
            write_gripper_targets_kernel,
            dim=self.world_count,
            inputs=[
                self.joint_q_ik,
                self.gripper_target,
                self.joint_targets_2d,
                self.arm_dof_count,
                self.gripper_dof_start,
                self.gripper_dof_count,
            ],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_2d.flatten())

        # Write arm + gripper to mujoco ctrl (CTRL_DIRECT for all MJCF general actuators)
        if self.has_mujoco_ctrl:
            wp.launch(
                write_mujoco_ctrl_kernel,
                dim=self.world_count,
                inputs=[
                    self.joint_q_ik,
                    self.gripper_target,
                    self.mujoco_ctrl_2d,
                    self.arm_dof_count,
                    self.gripper_actuator_idx,
                ],
            )

        # 5. Extract actual EE positions from simulation state
        wp.launch(
            extract_ee_pos_kernel,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.ee_body_global_indices,
            ],
            outputs=[
                self.ee_pos_actual,
            ],
        )

        # 6. Advance state machine (snapshots body_q on transition)
        bodies_per_world = self.model.body_count // self.world_count
        wp.launch(
            advance_task_kernel,
            dim=self.world_count,
            inputs=[
                self.task_idx,
                self.task_timer,
                self.task_durations,
                self.ee_pos_target,
                self.ee_pos_actual,
                self.state_0.body_q,
                self.task_init_body_q,
                bodies_per_world,
                self.frame_dt,
            ],
        )


# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------


@dataclass
class _PerWorldGraspSpec:
    """Per-world materialization of ``GRASP_SPECS`` as parallel CPU/GPU SoA arrays.

    The CPU mirrors (``*_np``) are mutated by the GUI Apply path and then assigned
    to the matching GPU arrays via ``.assign()``. ``*_src`` are 1-element staging
    buffers reused by the single-slot Apply path so the per-frame Apply costs no
    new allocations.
    """

    offset_local: wp.array[wp.vec3]
    quat_local: wp.array[wp.quat]
    ctrl: wp.array[wp.float32]
    offset_local_np: np.ndarray
    quat_local_np: np.ndarray
    ctrl_np: np.ndarray
    offset_src: wp.array[wp.vec3]
    quat_src: wp.array[wp.quat]
    ctrl_src: wp.array[wp.float32]


def zero_init(shape, dtype: type = wp.float32):
    """Allocate a Warp array filled with zeros. Thin wrapper for readability at call sites."""
    return wp.zeros(shape, dtype=dtype)


def full_init(shape, value, dtype: type = wp.float32):
    """Allocate a Warp array filled with ``value``. Thin wrapper for readability at call sites."""
    return wp.full(shape, value, dtype=dtype)


def alloc_line_buffers(n: int):
    """Allocate a paired ``(begin, end)`` ``wp.vec3`` buffer for log_lines channels."""
    return wp.zeros(n, dtype=wp.vec3), wp.zeros(n, dtype=wp.vec3)


def _print_table(title, columns, rows, *, separator="="):
    """Render a right-aligned table with auto-sized columns.

    ``columns``: ``[(header, formatter), ...]`` where ``formatter(row) -> str``.
    ``rows``: any iterable; each element is passed to every formatter.
    """
    headers = [h for h, _ in columns]
    cells = [[fmt(row) for _, fmt in columns] for row in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line_len = sum(widths) + 2 * max(0, len(widths) - 1)
    print(f"\n{separator * line_len}\n  {title}\n{separator * line_len}")
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths, strict=True)))
    print("-" * line_len)
    for row_cells in cells:
        print("  ".join(c.rjust(w) for c, w in zip(row_cells, widths, strict=True)))


def margin_pct_to_ctrl(margin_pct: float, y_half_m: float, stroke_mm: float = 85.0) -> float:
    """Convert per-shape margin_pct (fraction of object y-width) to Robotiq [0, 255] control.

    ``ctrl = 255 * (1 - (y_width - 2*margin) / stroke)``, clamped to ``[0, 255]``.
    Default stroke is the Robotiq 2F-85 full opening (85 mm).
    """
    y_width_mm = 2.0 * y_half_m * 1000.0
    margin_mm = margin_pct * y_width_mm
    ctrl = 255.0 * (1.0 - (y_width_mm - 2.0 * margin_mm) / stroke_mm)
    return min(255.0, max(0.0, ctrl))


def _quat_to_euler_zyx_deg(q: wp.quat) -> wp.vec3:
    """Convert a wp.quat to ZYX intrinsic Euler angles in degrees."""
    return 180.0 / wp.pi * wp.quat_to_rpy(q)


def _euler_zyx_deg_to_quat(rpy_deg: wp.vec3) -> wp.quat:
    """Convert ZYX intrinsic Euler angles in degrees to a wp.quat."""
    return wp.quat_from_euler(wp.pi / 180.0 * rpy_deg, 0, 1, 2)


def derive_offset_local_z(
    shape: ObjectShape,
    half_size: float,
    z_half: float,
    grasp_clearance: float = _GRASP_CLEARANCE,
) -> float:
    """Seed offset_local.z so the EE starts at spawn_center + grasp_clearance for the shape."""
    extra = _GRASP_Z_EXTRA.get(shape, 0.0)
    return (_GRASP_FLOOR_OFFSET_M + max(0.0, 2.0 * z_half - grasp_clearance) - z_half + extra) / half_size


# Builders for the primitive (non-mesh-asset) object shapes. Each entry takes the
# per-world half-size and returns a ``newton.Mesh`` -- we convert primitives to
# meshes so every world ends up with the same shape type, which the MuJoCo
# solver requires in ``separate_worlds=True`` mode.
_PRIMITIVE_MESH_FACTORIES = {
    ObjectShape.BOX: lambda hs: newton.Mesh.create_box(hs, hs, hs, compute_inertia=False),
    ObjectShape.SPHERE: lambda hs: newton.Mesh.create_sphere(hs, compute_inertia=False),
    ObjectShape.CYLINDER: lambda hs: newton.Mesh.create_cylinder(hs, hs, up_axis=newton.Axis.X, compute_inertia=False),
    ObjectShape.CAPSULE: lambda hs: newton.Mesh.create_capsule(hs, hs, up_axis=newton.Axis.X, compute_inertia=False),
    ObjectShape.ELLIPSOID: lambda hs: newton.Mesh.create_ellipsoid(hs * 2.0, hs, hs, compute_inertia=False),
}


# ----------------------------------------------------------------------------
# Warp kernels: grasp targets
# ----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def compute_grasp_targets(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    world_hs: wp.array[wp.float32],
    spec_offset_local: wp.array[wp.vec3],
    spec_quat_local: wp.array[wp.quat],
    spec_ctrl: wp.array[wp.float32],
    base_ee_rot: wp.quat,
    # outputs
    grasp_pos: wp.array[wp.vec3],
    grasp_rot: wp.array[wp.quat],
    grasp_ctrl: wp.array[wp.float32],
):
    """Compute world-frame grasp target from per-shape COM-frame spec.

    Rotation composition: ``body_q.q * base_ee_rot * spec_quat_local[w]``.
    """
    w = wp.tid()
    obj_global = body_world_start[w] + object_body_offset
    x_wb = body_q[obj_global]
    body_tr = wp.transform_get_translation(x_wb)
    body_q_rot = wp.transform_get_rotation(x_wb)

    com_local = body_com[obj_global]
    com_world = body_tr + wp.quat_rotate(body_q_rot, com_local)

    hs_w = world_hs[w]
    offset_local = spec_offset_local[w] * hs_w
    offset_world = wp.quat_rotate(body_q_rot, offset_local)

    grasp_pos[w] = com_world + offset_world
    grasp_rot[w] = body_q_rot * base_ee_rot * spec_quat_local[w]
    grasp_ctrl[w] = spec_ctrl[w]


@wp.kernel(enable_backward=False)
def compute_grasp_targets_slot(
    world_id: wp.int32,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    world_hs: wp.array[wp.float32],
    spec_offset_local: wp.array[wp.vec3],
    spec_quat_local: wp.array[wp.quat],
    spec_ctrl: wp.array[wp.float32],
    base_ee_rot: wp.quat,
    # outputs (only slot world_id is written)
    grasp_pos: wp.array[wp.vec3],
    grasp_rot: wp.array[wp.quat],
    grasp_ctrl: wp.array[wp.float32],
):
    """Single-slot variant of compute_grasp_targets. Launched with dim=1."""
    w = world_id
    obj_global = body_world_start[w] + object_body_offset
    x_wb = body_q[obj_global]
    body_tr = wp.transform_get_translation(x_wb)
    body_q_rot = wp.transform_get_rotation(x_wb)
    com_local = body_com[obj_global]
    com_world = body_tr + wp.quat_rotate(body_q_rot, com_local)
    hs_w = world_hs[w]
    offset_world = wp.quat_rotate(body_q_rot, spec_offset_local[w] * hs_w)
    grasp_pos[w] = com_world + offset_world
    grasp_rot[w] = body_q_rot * base_ee_rot * spec_quat_local[w]
    grasp_ctrl[w] = spec_ctrl[w]


# ----------------------------------------------------------------------------
# Warp kernels: state machine
# ----------------------------------------------------------------------------


@wp.func
def s_curve_profile(t: float, T: float, ramp_fraction: float) -> float:
    """S-curve trapezoidal position profile, normalized to [0, 1]."""
    t = wp.clamp(t, 0.0, T)
    f = wp.clamp(ramp_fraction, 0.01, 0.5)
    t_r = f * T
    v_max = 1.0 / (T - t_r)

    if t < t_r:
        return v_max * (t * 0.5 - t_r / (2.0 * wp.pi) * wp.sin(wp.pi * t / t_r))
    elif t < T - t_r:
        p1_end = v_max * t_r * 0.5
        return p1_end + v_max * (t - t_r)
    else:
        t_decel = t - (T - t_r)
        p12_end = v_max * t_r * 0.5 + v_max * (T - 2.0 * t_r)
        return p12_end + v_max * (t_decel * 0.5 + t_r / (2.0 * wp.pi) * wp.sin(wp.pi * t_decel / t_r))


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_idx: wp.array[wp.int32],
    task_timer: wp.array[wp.float32],
    task_durations: wp.array[wp.float32],
    grasp_pos: wp.array[wp.vec3],
    grasp_rot: wp.array[wp.quat],
    grasp_ctrl: wp.array[wp.float32],
    lift_distance_m: wp.float32,
    task_init_body_q: wp.array[wp.transform],
    ee_body_global_indices: wp.array[wp.int32],
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_pos_target_interp: wp.array[wp.vec3],
    ee_rot_target_interp: wp.array[wp.vec4],
    gripper_target: wp.array[wp.float32],
):
    """Compute IK target pose with interpolation from task start pose."""
    tid = wp.tid()
    state = task_idx[tid]
    timer = task_timer[tid]

    gp = grasp_pos[tid]
    gr = grasp_rot[tid]
    gc = grasp_ctrl[tid]
    lift_vec = wp.vec3(0.0, 0.0, lift_distance_m)

    if state == wp.static(int(TaskType.APPROACH)):
        target_pos = gp
        ctrl = 0.0
    elif state == wp.static(int(TaskType.CLOSE_GRIPPER)):
        target_pos = gp
        dur_close = task_durations[wp.static(int(TaskType.CLOSE_GRIPPER))]
        alpha = wp.clamp(timer / dur_close, 0.0, 1.0)
        ctrl = alpha * gc
    elif state == wp.static(int(TaskType.SETTLE)):
        target_pos = gp
        ctrl = gc
    elif state == wp.static(int(TaskType.LIFT)):
        target_pos = gp + lift_vec
        ctrl = gc
    elif state == wp.static(int(TaskType.HOLD)):
        target_pos = gp + lift_vec
        ctrl = gc
    else:
        target_pos = gp + lift_vec
        ctrl = gc

    ee_pos_target[tid] = target_pos
    # rot target: wp.vec4 because downstream consumers still take vec4
    ee_rot_target[tid] = wp.vec4(gr[0], gr[1], gr[2], gr[3])
    gripper_target[tid] = ctrl

    dur = task_durations[state] if state < wp.static(int(TaskType.DONE)) else 1.0
    t = wp.clamp(timer / dur, 0.0, 1.0)

    ee_body_idx = ee_body_global_indices[tid]
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_idx])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_idx])
    tcp_offset_local = wp.vec3(0.0, 0.0, wp.static(_ROBOTIQ_TCP_OFFSET_M))
    tcp_pos_prev = ee_pos_prev + wp.quat_rotate(ee_quat_prev, tcp_offset_local)

    ee_pos_target_interp[tid] = tcp_pos_prev * (1.0 - t) + target_pos * t
    ee_quat_interp = wp.quat_slerp(ee_quat_prev, gr, t)
    ee_rot_target_interp[tid] = wp.vec4(ee_quat_interp[0], ee_quat_interp[1], ee_quat_interp[2], ee_quat_interp[3])


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_idx: wp.array[wp.int32],
    task_timer: wp.array[wp.float32],
    task_durations: wp.array[wp.float32],
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_actual: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    task_init_body_q: wp.array[wp.transform],
    body_count: int,
    frame_dt: float,
):
    """Advance the per-world state machine. On transition, snapshot body_q."""
    tid = wp.tid()
    state = task_idx[tid]

    if state >= wp.static(int(TaskType.DONE)):
        return

    task_timer[tid] = task_timer[tid] + frame_dt

    dur = task_durations[state]
    timer = task_timer[tid]

    if timer < dur:
        return

    # For APPROACH and LIFT, also require EE position settling
    if state == wp.static(int(TaskType.APPROACH)) or state == wp.static(int(TaskType.LIFT)):
        target = ee_pos_target[tid]
        actual = ee_pos_actual[tid]
        err_x = wp.abs(target[0] - actual[0])
        err_y = wp.abs(target[1] - actual[1])
        err_z = wp.abs(target[2] - actual[2])
        settled = err_x < 0.001 and err_y < 0.001 and err_z < 0.00075
        if not settled:
            return

    # Advance to next state
    task_idx[tid] = state + 1
    task_timer[tid] = 0.0

    # Snapshot current body_q as start pose for the next task's interpolation
    body_start = tid * body_count
    for i in range(body_count):
        task_init_body_q[body_start + i] = body_q[body_start + i]


@wp.kernel(enable_backward=False)
def extract_ee_pos_kernel(
    body_q: wp.array[wp.transform],
    ee_body_global_indices: wp.array[wp.int32],
    # output
    ee_pos_actual: wp.array[wp.vec3],
):
    """Extract world-frame TCP position per world from body_q."""
    tid = wp.tid()
    body_idx = ee_body_global_indices[tid]
    ee_pos = wp.transform_get_translation(body_q[body_idx])
    ee_quat = wp.transform_get_rotation(body_q[body_idx])
    tcp_offset_local = wp.vec3(0.0, 0.0, wp.static(_ROBOTIQ_TCP_OFFSET_M))
    ee_pos_actual[tid] = ee_pos + wp.quat_rotate(ee_quat, tcp_offset_local)


@wp.kernel(enable_backward=False)
def write_gripper_targets_kernel(
    ik_solution: wp.array2d[wp.float32],
    gripper_target: wp.array[wp.float32],
    joint_targets: wp.array2d[wp.float32],
    arm_dof_count: int,
    gripper_dof_start: int,
    gripper_dof_count: int,
):
    """Copy IK arm solution + gripper target to joint_targets 2D array."""
    tid = wp.tid()

    # Copy arm DOFs from IK solution
    for j in range(arm_dof_count):
        joint_targets[tid, j] = ik_solution[tid, j]

    # Write gripper DOFs: all gripper joints get the same control value
    ctrl = gripper_target[tid]
    for j in range(gripper_dof_count):
        joint_targets[tid, gripper_dof_start + j] = ctrl


@wp.kernel(enable_backward=False)
def write_mujoco_ctrl_kernel(
    ik_solution: wp.array2d[wp.float32],
    gripper_target: wp.array[wp.float32],
    mujoco_ctrl: wp.array2d[wp.float32],
    arm_actuator_count: int,
    gripper_actuator_idx: int,
):
    """Write IK arm solution + gripper target to MuJoCo ctrl."""
    tid = wp.tid()
    for j in range(arm_actuator_count):
        mujoco_ctrl[tid, j] = ik_solution[tid, j]
    mujoco_ctrl[tid, gripper_actuator_idx] = gripper_target[tid]


# GUI staging buffer indices (keep in sync with stage_gui_metrics_kernel in
# newton.tests.test_object_centric_grasp.GraspProbe).
_GUI_STAGE_FIELDS = [
    "pad_force_cur",
    "pad_force_max",
    "pad_friction_cur",
    "pad_friction_max",
    "table_force_cur",
    "table_force_max",
    "table_friction_cur",
    "table_friction_max",
    "penetration_cur",
    "penetration_max",
    "avg_vel",
    "object_z",
    "object_z_init",
    "object_z_max",
    "task_idx",  # stored as float for uniform array
    "world_nan_frame",  # -1 sentinel -> NaN semantics preserved via float cast
    "task_timer",  # task_timer[selected_world]
    "task_dur",  # task_durations[cur_task], or 0 if cur_task >= task_duration_count
    "ee_target_x",
    "ee_target_y",
    "ee_target_z",
    "ee_actual_x",
    "ee_actual_y",
    "ee_actual_z",
]
_GUI_STAGE_SIZE = len(_GUI_STAGE_FIELDS)

# ----------------------------------------------------------------------------
# Asset / geometry helpers
# ----------------------------------------------------------------------------


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


# ---- LEGO brick mesh generation ----


def _lego_cylinder_mesh(radius, height, segments, cx=0.0, cy=0.0, cz=0.0, bottom_cap=True):
    n = segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    ring_x = cx + radius * cos_a
    ring_y = cy + radius * sin_a
    verts = []
    faces = []
    side_bot = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
    side_top = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    verts.append(side_bot)
    verts.append(side_top)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + j, n + i])
        faces.append([i, j, n + j])
    off_top = 2 * n
    cap_top_ring = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    cap_top_center = np.array([[cx, cy, cz + height]], dtype=np.float32)
    verts.append(cap_top_ring)
    verts.append(cap_top_center)
    tc = off_top + n
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc, off_top + i, off_top + j])
    if bottom_cap:
        off_bot = off_top + n + 1
        cap_bot_ring = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
        cap_bot_center = np.array([[cx, cy, cz]], dtype=np.float32)
        verts.append(cap_bot_ring)
        verts.append(cap_bot_center)
        bc = off_bot + n
        for i in range(n):
            j = (i + 1) % n
            faces.append([bc, off_bot + j, off_bot + i])
    return np.vstack(verts), np.array(faces, dtype=np.int32)


def _lego_combine_meshes(mesh_list):
    all_v, all_f, off = [], [], 0
    for v, f in mesh_list:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v).astype(np.float32), np.vstack(all_f).astype(np.int32)


def _lego_make_shell_mesh(nx, ny):
    ox = nx * _LEGO_PITCH / 2.0
    oy = ny * _LEGO_PITCH / 2.0
    inx = ox - _LEGO_WALL_THICKNESS
    iny = oy - _LEGO_WALL_THICKNESS
    H = _LEGO_BRICK_HEIGHT
    T = _LEGO_TOP_THICKNESS
    v = np.array(
        [
            [-ox, -oy, 0],
            [+ox, -oy, 0],
            [+ox, +oy, 0],
            [-ox, +oy, 0],
            [-ox, -oy, H],
            [+ox, -oy, H],
            [+ox, +oy, H],
            [-ox, +oy, H],
            [-inx, -iny, 0],
            [+inx, -iny, 0],
            [+inx, +iny, 0],
            [-inx, +iny, 0],
            [-inx, -iny, H - T],
            [+inx, -iny, H - T],
            [+inx, +iny, H - T],
            [-inx, +iny, H - T],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [1, 2, 6],
            [1, 6, 5],
            [0, 8, 9],
            [0, 9, 1],
            [1, 9, 10],
            [1, 10, 2],
            [2, 10, 11],
            [2, 11, 3],
            [3, 11, 8],
            [3, 8, 0],
            [9, 8, 12],
            [9, 12, 13],
            [11, 10, 14],
            [11, 14, 15],
            [8, 11, 15],
            [8, 15, 12],
            [10, 9, 13],
            [10, 13, 14],
            [12, 15, 14],
            [12, 14, 13],
        ],
        dtype=np.int32,
    )
    return v, f


def _lego_make_brick_mesh(nx=4, ny=2):
    shell_v, shell_f = _lego_make_shell_mesh(nx, ny)
    seg = _LEGO_CYLINDER_SEGMENTS
    stud_meshes = []
    for i in range(nx):
        for j in range(ny):
            sx = (i - (nx - 1) / 2.0) * _LEGO_PITCH
            sy = (j - (ny - 1) / 2.0) * _LEGO_PITCH
            stud_meshes.append(
                _lego_cylinder_mesh(
                    _LEGO_STUD_RADIUS,
                    _LEGO_STUD_HEIGHT,
                    seg,
                    cx=sx,
                    cy=sy,
                    cz=_LEGO_BRICK_HEIGHT,
                    bottom_cap=False,
                )
            )
    tube_meshes = []
    if ny == 2:
        tube_height = _LEGO_BRICK_HEIGHT - _LEGO_TOP_THICKNESS
        for i in range(nx - 1):
            tx = (i - (nx - 2) / 2.0) * _LEGO_PITCH
            tube_meshes.append(_lego_cylinder_mesh(_LEGO_TUBE_OUTER_RADIUS, tube_height, seg, cx=tx, cy=0.0, cz=0.0))
    v, f = _lego_combine_meshes([(shell_v, shell_f), *stud_meshes, *tube_meshes])
    center = (v.min(axis=0) + v.max(axis=0)) / 2.0
    v -= center
    return newton.Mesh(v, f.flatten())


# ---- Mesh loading helpers for local USD and external OBJ assets ----


def _load_usd_mesh_local(usd_path, prim_path):
    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(prim_path)
    mesh = newton.usd.get_mesh(prim, load_normals=False)
    verts = mesh.vertices
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    if not np.allclose(center, 0.0, atol=1e-4):
        mesh = mesh.copy(vertices=verts - center, recompute_inertia=True)
    return mesh


def _load_obj_mesh_trimesh(obj_path):
    import trimesh  # noqa: PLC0415

    tm = trimesh.load(obj_path, force="mesh")
    vertices = np.array(tm.vertices, dtype=np.float32)
    indices = np.array(tm.faces.flatten(), dtype=np.int32)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices -= center
    return newton.Mesh(vertices, indices)


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
