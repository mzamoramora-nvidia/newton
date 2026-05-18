# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Heterogeneous Grasp
#
# Demonstrates heterogeneous grasping environments: each world contains a
# different object (shape, mass, size), all grasped and lifted by a Franka
# Panda + Robotiq 2F-85 gripper. Selectable SDF / hydroelastic collisions.
#
# Command: python -m newton.examples robot_heterogeneous_grasp
#
###########################################################################

import copy
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

_BRICK_PITCH = 0.008
_BRICK_HEIGHT = 0.0096
_BRICK_STUD_RADIUS = 0.0024
_BRICK_STUD_HEIGHT = 0.0017
_BRICK_WALL_THICKNESS = 0.0012
_BRICK_TOP_THICKNESS = 0.001
_BRICK_TUBE_OUTER_RADIUS = 0.003255
_BRICK_CYLINDER_SEGMENTS = 24

# Table half-extents [m]. Sized to support the Franka base at X = -0.5 plus
# the spawn region at X ~= 0. If you change these, update the table SDF
# resolution in _setup_collision_sdf to keep the voxel size near 5 mm.
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

# Per-world spawn-pose randomization. Both ranges are uniform [-r, +r]; the
# seed (CLI arg) controls determinism.
_SPAWN_XY_RANGE_M = 0.10
_SPAWN_YAW_RANGE_DEG = 30.0

# Hydroelastic stiffness applied to object meshes and gripper pads. [Pa]
_HYDROELASTIC_KH_PA = 2e11

# Simulation rate: sim_dt = frame_dt / _SUBSTEPS_PER_FRAME. Collision pipeline
# runs every _COLLIDE_EVERY_N_SUBSTEPS substeps.
_SUBSTEPS_PER_FRAME = 16
_COLLIDE_EVERY_N_SUBSTEPS = 4


class ObjectShape(IntEnum):
    BOX = 0
    SPHERE = 1
    CYLINDER = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CUP = 5
    RUBBER_DUCK = 6
    BRICK = 7
    RJ45_PLUG = 8
    BEAR = 9
    NUT = 10
    BOLT = 11


SHAPE_NAMES = [s.name for s in ObjectShape]
NUM_SHAPES = len(ObjectShape)

_MESH_SHAPES = {
    ObjectShape.CUP,
    ObjectShape.RUBBER_DUCK,
    ObjectShape.BRICK,
    ObjectShape.RJ45_PLUG,
    ObjectShape.BEAR,
    ObjectShape.NUT,
    ObjectShape.BOLT,
}


class CollisionMode(IntEnum):
    NEWTON_SDF = 0
    NEWTON_HYDROELASTIC = 1


class TaskType(IntEnum):
    APPROACH = 0
    CLOSE_GRIPPER = 1
    SETTLE = 2  # hold pose + full closure; lets contact forces stabilize before LIFT
    LIFT = 3
    HOLD = 4
    DONE = 5


NUM_TASKS = len(TaskType)
TASK_NAMES = [t.name for t in TaskType]


# Default depth (m) the EE seed sits below the object's top -- i.e. how far
# the fingers extend past the top surface to grip the body. Clamped against
# the table for thin objects in derive_pos_offset_z.
_GRASP_DEPTH_FROM_TOP = 0.05


@dataclass(frozen=True)
class GraspSpec:
    """Per-shape grasp pose, authored in the object's COM frame.

    The position target is composed of two vec3s with different unit
    conventions, summed at runtime:

    - ``pos_offset_fractional`` is in units of the per-world object half-size.
      The kernel multiplies it by ``half_size`` to recover meters, so the
      same spec drives the gripper to the same relative pose regardless of
      object size.
    - ``pos_offset_absolute`` is in absolute world meters. Use it for fixed
      lifts (e.g. extra headroom for tall / awkward shapes) that should not
      scale with the per-world object size.

    ``quat_offset`` rotates the EE target around the body, on top of the
    shared ``base_ee_rot``.

    ``overclose_fraction`` controls the gripper closure: each pad closes
    past the object surface by this fraction of the object's full Y-width.
    A value of 0 just touches the surface; 0.05 squeezes 5% of the width
    deeper per side.

    Composed at runtime as:
        pos_offset_m   = pos_offset_fractional * half_size + pos_offset_absolute
        grasp_pos_world = com_world + body_q.q * pos_offset_m
        grasp_rot_world = body_q.q * base_ee_rot * quat_offset
        grasp_ctrl      = overclose_to_ctrl(overclose_fraction, y_half)
    """

    pos_offset_fractional: wp.vec3 = field(default_factory=lambda: wp.vec3(0.0, 0.0, 0.0))
    pos_offset_absolute: wp.vec3 = field(default_factory=lambda: wp.vec3(0.0, 0.0, 0.0))
    quat_offset: wp.quat = field(default_factory=wp.quat_identity)
    overclose_fraction: float = 0.05


GRASP_SPECS: dict[ObjectShape, GraspSpec] = {
    ObjectShape.BOX: GraspSpec(overclose_fraction=0.05),
    ObjectShape.SPHERE: GraspSpec(overclose_fraction=0.05),
    ObjectShape.CYLINDER: GraspSpec(overclose_fraction=0.05),
    ObjectShape.CAPSULE: GraspSpec(overclose_fraction=0.05),
    ObjectShape.ELLIPSOID: GraspSpec(overclose_fraction=0.05),
    ObjectShape.CUP: GraspSpec(overclose_fraction=0.22),
    ObjectShape.RUBBER_DUCK: GraspSpec(overclose_fraction=0.10),
    ObjectShape.BRICK: GraspSpec(overclose_fraction=0.15),
    ObjectShape.RJ45_PLUG: GraspSpec(
        pos_offset_fractional=wp.vec3(0.0, 0.30, 0.0),
        quat_offset=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 90.0 * wp.pi / 180.0),
        overclose_fraction=0.10,
    ),
    ObjectShape.BEAR: GraspSpec(overclose_fraction=0.25, pos_offset_absolute=wp.vec3(0.0, 0.0, 0.01)),
    ObjectShape.NUT: GraspSpec(
        quat_offset=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 30.0 * wp.pi / 180.0),
        overclose_fraction=0.15,
    ),
    ObjectShape.BOLT: GraspSpec(overclose_fraction=0.30, pos_offset_absolute=wp.vec3(0.0, 0.0, 0.02)),
}


class Example:
    def __init__(self, viewer, args):
        self.test_mode = args.test
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = _SUBSTEPS_PER_FRAME
        self.collide_substeps = _COLLIDE_EVERY_N_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = args.world_count
        self.seed = args.seed
        self.collision_mode = CollisionMode[args.collision_mode.upper()]
        self.verbose = args.verbose
        self.viewer = viewer
        self.episode_steps = 0

        self._generate_world_params()
        robot_builder, arm_only_builder = self._build_robot()
        self.model_arm_only = arm_only_builder.finalize()
        self._load_mesh_objects()
        scene = self._build_scene(robot_builder)
        self._setup_collision_sdf(scene)
        self.model = scene.finalize()

        self.rigid_contact_max = 2_000 * self.world_count
        print(
            f"Bodies: {self.model.body_count}, Joints: {self.model.joint_count}, DOFs: {self.model.joint_coord_count}"
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._setup_ik()
        self._setup_state_machine()
        self.control = self.model.control()
        self.joint_target_shape = self.control.joint_target_pos.reshape((self.world_count, -1)).shape
        self.joint_targets_2d = wp.zeros(self.joint_target_shape, dtype=wp.float32)
        self.graph_ik = None

        # Direct-control path for the MJCF general actuators.
        ctrl = self.control.mujoco.ctrl
        actuator_count = ctrl.shape[0] // self.world_count
        self.mujoco_ctrl_2d = ctrl.reshape((self.world_count, actuator_count))
        self.gripper_actuator_idx = self.arm_dof_count  # gripper actuator follows the 7 arm actuators
        print(f"MuJoCo ctrl: {actuator_count} actuators/world, gripper at idx {self.gripper_actuator_idx}")

        # Seed ctrl with the initial arm joint positions so the arm holds its pose
        # on frame 0 before IK starts driving it.
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
        self.capture()

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
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
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
        self.sim_time += self.frame_dt
        self.episode_steps += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
            if self.collision_pipeline is not None and self.collision_pipeline.hydroelastic_sdf is not None:
                self.viewer.log_hydro_contact_surface(
                    self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
                    penetrating_only=True,
                )
        self.viewer.end_frame()

    def test_final(self):
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
            choices=["newton_sdf", "newton_hydroelastic"],
            default="newton_hydroelastic",
            help="Collision pipeline to use (SDF or hydroelastic).",
        )
        parser.add_argument("--seed", type=int, default=42, help="RNG seed for mass/size variation")
        parser.add_argument(
            "--verbose", action="store_true", help="Print per-world shape / body-COM diagnostics at startup."
        )
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
            kh=_HYDROELASTIC_KH_PA,
            gap=0.0005,
            mu=1.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg = self.shape_cfg
        builder.rigid_gap = self.shape_cfg.gap
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # Scene layout: robot 0.5 m behind the table center along -X, both on the world Y axis.
        self.table_pos = wp.vec3(-0.275, 0.0, _TABLE_HEIGHT / 2.0)
        self.spawn_center = wp.vec3(0.0, 0.0, _TABLE_HEIGHT)
        self.robot_base_pos = wp.vec3(-0.5, 0.0, _TABLE_HEIGHT)

        builder.add_mjcf(
            str(self._franka_dir / "panda_nohand.xml"),
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )

        # Reach-forward arm pose for the initial IK seed.
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

        # Snapshot before attaching the gripper -- IK runs on the arm chain only,
        # with link_offset mapping the EE link to the TCP.
        arm_only_builder = copy.deepcopy(builder)

        def find_body(name):
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        # Attach Robotiq 2F-85 V4 to link7 using the MJCF's attachment-body
        # transform plus an extra 90 deg Z rotation for better jaw alignment.
        # q_attach reorders the MJCF (w, x, y, z) quat to Warp's (x, y, z, w).
        q_attach = wp.quat(0.0, 0.0, 0.9238795, 0.3826834)
        q_90z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        builder.add_mjcf(
            str(self._robotiq_dir / "2f85.xml"),
            parent_body=find_body("link7"),
            xform=wp.transform(wp.vec3(0.0, 0.0, _PANDA_FLANGE_OFFSET_M), q_attach * q_90z),
            enable_self_collisions=False,
        )

        # The MJCF's attachment body is massless; give it a tiny inertia so
        # the validator doesn't warn.
        self.ee_attachment_body_idx = find_body("attachment")
        builder.body_mass[self.ee_attachment_body_idx] = 1e-6
        builder.body_inertia[self.ee_attachment_body_idx] = wp.mat33(1e-9, 0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 0.0, 1e-9)
        franka_body_count = self.ee_attachment_body_idx + 1
        self.ee_base_body_idx = next(
            i for i in range(franka_body_count, builder.body_count) if builder.body_label[i].endswith("/base")
        )

        # panda_nohand: 7 arm DOFs, no finger joints; gripper DOFs follow.
        self.arm_dof_count = 7
        self.gripper_dof_start = self.arm_dof_count  # Robotiq DOFs start right after arm
        self.gripper_dof_count = builder.joint_coord_count - self.arm_dof_count

        print(
            f"Joint layout: {self.arm_dof_count} arm + "
            f"{self.gripper_dof_count} gripper = {builder.joint_coord_count} total DOFs"
        )
        print(f"Gripper DOF range: [{self.gripper_dof_start}, {self.gripper_dof_start + self.gripper_dof_count})")

        # 2x the MJCF default armature -- the default is too small to stay
        # stable under hydroelastic contact forces. Order matches the 6 V4
        # gripper joints: driver, spring_link, follower per side.
        gripper_armature = [0.010, 0.002, 0.002, 0.010, 0.002, 0.002]
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

        # Round-robin shape assignment across all ObjectShape values
        self.world_shapes = [ObjectShape(i % NUM_SHAPES) for i in range(n)]

        # Fixed density (1000 kg/m^3 ~= water): mass scales with shape volume and size,
        # so we don't have to combine an unrealistic density with the per-world size jitter.
        self.object_density = 1000.0

        # Per-world half-extents as a flat (n, 3) array: (x_half, y_half, z_half).
        # x_half is drawn here with +/- 25% uniform jitter around the base; y_half
        # and z_half are filled by _add_object once the mesh is finalized.
        base_half_size = 0.025
        self.world_half_sizes = np.zeros((n, 3), dtype=np.float32)
        self.world_half_sizes[:, 0] = base_half_size * rng.uniform(0.75, 1.25, size=n)

        # Per-world spawn pose randomization: XY offset on the table and Z-yaw rotation.
        # Both draws are uniform in [-range, +range]; ranges of 0 collapse to deterministic
        # spawns at the table center with the per-shape default orientation.
        spawn_yaw_range_rad = _SPAWN_YAW_RANGE_DEG * wp.pi / 180.0
        self._world_spawn_xy = rng.uniform(-_SPAWN_XY_RANGE_M, _SPAWN_XY_RANGE_M, size=(n, 2)).astype(np.float32)
        self._world_spawn_yaw = rng.uniform(-spawn_yaw_range_rad, spawn_yaw_range_rad, size=n).astype(np.float32)

        if self.verbose:
            for i in range(n):
                print(
                    f"  World {i:3d}: shape={SHAPE_NAMES[self.world_shapes[i]]:>12s}  "
                    f"hs={self.world_half_sizes[i, 0] * 1000:.1f} mm"
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

        if ObjectShape.BRICK in needed:
            self.mesh_objects[ObjectShape.BRICK] = _make_brick_mesh(4, 2)

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
        half_size = float(self.world_half_sizes[world_id, 0])
        mesh = self.mesh_objects[shape] if shape in _MESH_SHAPES else _PRIMITIVE_MESH_FACTORIES[shape](half_size)

        if shape in _MESH_SHAPES:
            extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
            uniform_scale = 2.0 * half_size / extents.max() if extents.max() > 0 else 1.0
            # RJ45_PLUG is spawned with a 90° Z-yaw so its body-frame Y is the mesh's X-extent.
            y_extent = extents[0] if shape == ObjectShape.RJ45_PLUG else extents[1]
            self.world_half_sizes[world_id, 1] = y_extent / 2.0 * uniform_scale
            self.world_half_sizes[world_id, 2] = extents[2] / 2.0 * uniform_scale
        else:
            uniform_scale = 1.0
            self.world_half_sizes[world_id, 1] = half_size
            self.world_half_sizes[world_id, 2] = half_size
        scale = wp.vec3(uniform_scale, uniform_scale, uniform_scale)

        z_axis = wp.vec3(0.0, 0.0, 1.0)
        obj_rot = wp.quat_from_axis_angle(z_axis, wp.pi / 2.0) if shape == ObjectShape.RJ45_PLUG else wp.quat_identity()
        # Per-world spawn randomization: XY offset + Z-yaw composed onto obj_rot.
        sx, sy, sz = self.spawn_center
        dx, dy = self._world_spawn_xy[world_id]
        yaw_rot = wp.quat_from_axis_angle(z_axis, self._world_spawn_yaw[world_id])
        obj_z = sz + self.world_half_sizes[world_id, 2] + _GRASP_FLOOR_OFFSET_M
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

        # The brick gets explicit floor + four wall colliders inside its hollow shell;
        # other shapes get five tiny placeholder boxes so every world has identical body
        # topology (required by MuJoCo's separate_worlds=True mode).
        collider_cfg = replace(self.shape_cfg, density=0.0, is_visible=False)
        if shape == ObjectShape.BRICK:
            sf = uniform_scale
            inset = 0.0001 * sf
            ox = _BRICK_PITCH * 2 * sf
            oy = _BRICK_PITCH * sf
            center_z = (_BRICK_HEIGHT + _BRICK_STUD_HEIGHT) / 2.0 * sf
            box_hz = 0.5 * _BRICK_HEIGHT * sf - inset
            box_cz = 0.5 * _BRICK_HEIGHT * sf - center_z
            wt = _BRICK_WALL_THICKNESS * sf
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
        # Kinematic per-world table body; static (body=-1) doesn't replicate.
        table_mesh = newton.Mesh.create_box(_TABLE_HALF_X, _TABLE_HALF_Y, _TABLE_HEIGHT / 2.0, compute_inertia=False)
        table_cfg = replace(self.shape_cfg, density=0.0)

        scene = newton.ModelBuilder()
        table_shapes: list[int] = []
        for world_id in range(self.world_count):
            scene.begin_world()
            link0_body = scene.body_count  # first body added by add_builder is the panda link0
            scene.add_builder(robot_builder)

            table_body = scene.add_body(
                xform=wp.transform(self.table_pos, wp.quat_identity()),
                label="table",
                is_kinematic=True,
            )
            table_shape = scene.add_shape_mesh(
                body=table_body,
                mesh=table_mesh,
                cfg=table_cfg,
                color=(0.20, 0.20, 0.22),
                label="table_shape",
            )
            table_shapes.append(table_shape)

            # The robot base rests on the table top; filter link0 vs table to
            # avoid wasted contact generation on a pair that can never separate.
            for shape_idx in scene.body_shapes[link0_body]:
                scene.add_shape_collision_filter_pair(shape_idx, table_shape)

            obj_body = self._add_object(scene, world_id)

            if world_id == 0:
                self.object_body_offset = obj_body

            scene.end_world()

        ground_shape = scene.add_ground_plane()
        # Table bottom sits on the ground plane; filter every per-world table
        # against the single shared ground shape.
        for table_shape in table_shapes:
            scene.add_shape_collision_filter_pair(table_shape, ground_shape)
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
        init_ee_rot = wp.transform_get_rotation(ee_tf)

        # link_offset maps IK targets to the TCP. The initial position target is
        # also the current TCP so the seed is already a perfect solution and IK
        # doesn't drift on frame 0.
        tcp_offset_vec = wp.vec3(0.0, 0.0, _ROBOTIQ_TCP_OFFSET_M)
        init_tcp_pos = wp.transform_point(ee_tf, tcp_offset_vec)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_index,
            link_offset=tcp_offset_vec,
            target_positions=wp.array([init_tcp_pos] * self.world_count, dtype=wp.vec3),
        )

        # Hold the attachment body at its initial orientation -- the IK runs on
        # the arm chain only; the Robotiq's extra 90 deg Z mount is baked into
        # base_ee_rot downstream.
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

        # Arm-only attachment-body rotation; same source the IK rotation
        # objective uses so the grasp composition stays consistent.
        state_arm = self.model_arm_only.state()
        newton.eval_fk(self.model_arm_only, self.model_arm_only.joint_q, self.model_arm_only.joint_qd, state_arm)
        arm_ee_rot = wp.transform_get_rotation(wp.transform(*state_arm.body_q.numpy()[self.ik_ee_index]))

        # Build the per-world spec arrays from GRASP_SPECS.
        frac_arr = np.zeros((self.world_count, 3), dtype=np.float32)
        abs_arr = np.zeros((self.world_count, 3), dtype=np.float32)
        quat_arr = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (self.world_count, 1))
        ctrl_arr = np.zeros(self.world_count, dtype=np.float32)
        for i, shape in enumerate(self.world_shapes):
            spec = GRASP_SPECS[shape]
            _x_half, y_half, z_half = self.world_half_sizes[i]
            frac_arr[i] = spec.pos_offset_fractional
            # Fold the auto-seeded Z offset (in meters) into the absolute Z.
            abs_arr[i] = (
                spec.pos_offset_absolute[0],
                spec.pos_offset_absolute[1],
                spec.pos_offset_absolute[2] + derive_pos_offset_z(z_half=z_half),
            )
            quat_arr[i] = spec.quat_offset
            ctrl_arr[i] = overclose_to_ctrl(spec.overclose_fraction, y_half_m=y_half)
        self.spec = _PerWorldGraspSpec(
            pos_offset_fractional=wp.array(frac_arr, dtype=wp.vec3),
            pos_offset_absolute=wp.array(abs_arr, dtype=wp.vec3),
            quat_offset=wp.array(quat_arr, dtype=wp.quat),
            ctrl=wp.array(ctrl_arr, dtype=wp.float32),
        )

        # GPU runtime SoA outputs of compute_grasp_targets.
        self.grasp_pos = wp.zeros(self.world_count, dtype=wp.vec3)
        self.grasp_rot = wp.zeros(self.world_count, dtype=wp.quat)
        self.grasp_ctrl = wp.zeros(self.world_count, dtype=wp.float32)

        self._world_half_size_array = wp.array(self.world_half_sizes[:, 0].copy(), dtype=wp.float32)
        self.base_ee_rot = wp.quat(*arm_ee_rot)

        if self.verbose:
            body_com_np = self.model.body_com.numpy()
            body_ws_np = self.model.body_world_start.numpy()
            first_world = {shape: idx for idx, shape in reversed(list(enumerate(self.world_shapes)))}
            print("[grasp] per-shape body_com magnitudes (body-local frame):")
            for shape, idx in sorted(first_world.items(), key=lambda kv: kv[0].name):
                com_norm = np.linalg.norm(body_com_np[int(body_ws_np[idx]) + self.object_body_offset])
                flag = "  (non-zero, review pos_offset)" if com_norm > 1e-4 else ""
                print(f"  {shape.name:<12} |body_com| = {com_norm * 1000.0:8.3f} mm{flag}")

        self.lift_distance_m: float = 0.1

        wp.launch(
            compute_grasp_targets,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.model.body_com,
                self.model.body_world_start,
                self.object_body_offset,
                self._world_half_size_array,
                self.spec.pos_offset_fractional,
                self.spec.pos_offset_absolute,
                self.spec.quat_offset,
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

        # Global EE body indices per world (attachment body, matching the IK EE link).
        body_ws = self.model.body_world_start.numpy()[: self.world_count].astype(np.int32)
        self.ee_body_global_indices = wp.array(body_ws + self.ee_attachment_body_idx, dtype=wp.int32)

        # Snapshot body_q at each task start, used to interpolate the EE target.
        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.body_count_total = self.model.body_count

    def _setup_collision_sdf(self, builder):
        """Build SDFs on all collision shapes; mark fingertips, objects, and table hydroelastic.

        Operates on the scene builder BEFORE finalize():
        Pass 1 - SDF on every collision shape (BOX -> MESH + SDF build).
        Pass 2 - HYDROELASTIC flag on fingertip pads, object shapes, and the
        table (so the contact-surface visualization always has at least one
        pair to draw). Only applied in NEWTON_HYDROELASTIC mode.
        """
        sdf_narrow_band = (-0.0015, 0.0015)
        hydroelastic_enabled = self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC

        # ---- Pass 1: build SDF on every collision shape ----
        # Object shapes get higher resolution (96) for better contact quality;
        # robot links, gripper, and table use 64.
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
            # Object meshes get a smaller margin to avoid inflating thin features
            sdf_margin = 0.0002 if is_object else self.shape_cfg.gap

            if builder.shape_type[shape_idx] == newton.GeoType.BOX:
                # Convert BOX to MESH + SDF (needed for pad shapes from MJCF)
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
                    # Bake scale into vertices and rebuild SDF (required for hydroelastic)
                    mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
                    builder.shape_source[shape_idx] = mesh
                    builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    if mesh.sdf is not None:
                        mesh.clear_sdf()
                    mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)
                elif mesh.sdf is None:
                    mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)

        # ---- Pass 2: fingertip, object, and table shapes get hydroelastic flag ----
        if hydroelastic_enabled:
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
                    cfg = self.shape_cfg
                    builder.shape_gap[shape_idx] = cfg.gap
                    builder.shape_material_mu[shape_idx] = cfg.mu
                    builder.shape_material_mu_torsional[shape_idx] = cfg.mu_torsional
                    builder.shape_material_mu_rolling[shape_idx] = cfg.mu_rolling
                    if is_table:
                        builder.shape_material_kh[shape_idx] = _TABLE_KH_PA
                    else:
                        builder.shape_material_kh[shape_idx] = _HYDROELASTIC_KH_PA
                    builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
                    if is_fingertip:
                        builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.VISIBLE
                    hydro_count += 1

            print(f"[SDF setup] Marked {hydro_count} shapes as HYDROELASTIC")

    def _create_collision_pipeline(self):
        """Create the Newton collision pipeline (SDF or hydroelastic)."""
        if self.collision_mode == CollisionMode.NEWTON_SDF:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                broad_phase="explicit",
                reduce_contacts=True,
            )
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
        else:
            raise ValueError(f"Unknown collision mode: {self.collision_mode}")
        self.contacts = self.collision_pipeline.contacts()

    def _create_solver(self):
        """Create the MuJoCo solver (use_mujoco_contacts=False -- contacts come
        from the Newton collision pipeline)."""
        nconmax_per_world = self.rigid_contact_max // self.world_count
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=100,
            ls_iterations=200,
            impratio=50.0,
            njmax=nconmax_per_world,
            nconmax=nconmax_per_world,
        )

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
    """GPU-resident per-world materialization of ``GRASP_SPECS`` (SoA layout)."""

    pos_offset_fractional: wp.array[wp.vec3]
    pos_offset_absolute: wp.array[wp.vec3]
    quat_offset: wp.array[wp.quat]
    ctrl: wp.array[wp.float32]


def overclose_to_ctrl(overclose_fraction: float, y_half_m: float, stroke_mm: float = 85.0) -> float:
    """Convert a per-shape grip overclose fraction to a Robotiq [0, 255] control value.

    ``overclose_fraction`` is how far each pad closes past the object
    surface, expressed as a fraction of the object's full Y-width. The
    resulting opening between the pads is ``y_width * (1 - 2 * fraction)``,
    and ``ctrl`` is the corresponding closure level on the Robotiq 2F-85
    (0 = fully open, 255 = fully closed). Default stroke is the 2F-85 full
    opening (85 mm). Output is clamped to ``[0, 255]``.
    """
    y_width_mm = 2.0 * y_half_m * 1000.0
    overclose_mm = overclose_fraction * y_width_mm
    pad_opening_mm = y_width_mm - 2.0 * overclose_mm
    ctrl = 255.0 * (1.0 - pad_opening_mm / stroke_mm)
    return min(255.0, max(0.0, ctrl))


def derive_pos_offset_z(z_half: float, grasp_depth: float = _GRASP_DEPTH_FROM_TOP) -> float:
    """Seed Z offset (m) from the object COM where the EE should grasp.

    Geometry::

        EE seed         <- obj_center + return value
        ────────  obj_top
                 \\
                  ) z_half
                 /
        obj_center
                 \\
                  ) z_half
                 /
        ────────  obj_bottom
        ░░░░░░░░░░░░  table_top
            └ floor_gap = _GRASP_FLOOR_OFFSET_M

    The EE seed is placed at ``obj_top - grasp_depth`` (i.e. the fingers
    reach ``grasp_depth`` past the top surface) when the object is at least
    ``grasp_depth`` tall; otherwise it clamps to the table top so the
    gripper doesn't dip below it.
    """
    return _GRASP_FLOOR_OFFSET_M + max(0.0, 2.0 * z_half - grasp_depth) - z_half


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
    spec_pos_offset_fractional: wp.array[wp.vec3],
    spec_pos_offset_absolute: wp.array[wp.vec3],
    spec_quat_offset: wp.array[wp.quat],
    spec_ctrl: wp.array[wp.float32],
    base_ee_rot: wp.quat,
    # outputs
    grasp_pos: wp.array[wp.vec3],
    grasp_rot: wp.array[wp.quat],
    grasp_ctrl: wp.array[wp.float32],
):
    """Compute world-frame grasp target from per-shape COM-frame spec.

    Position composition: ``com + body_q.q * (frac * half_size + absolute)``.
    Rotation composition: ``body_q.q * base_ee_rot * spec_quat_offset[w]``.
    """
    w = wp.tid()
    obj_global = body_world_start[w] + object_body_offset
    x_wb = body_q[obj_global]

    com_local = body_com[obj_global]
    hs_w = world_hs[w]
    pos_offset = spec_pos_offset_fractional[w] * hs_w + spec_pos_offset_absolute[w]

    grasp_pos[w] = wp.transform_point(x_wb, com_local + pos_offset)
    grasp_rot[w] = wp.transform_get_rotation(x_wb) * base_ee_rot * spec_quat_offset[w]
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
    ee_init_tf = task_init_body_q[ee_body_idx]
    ee_quat_prev = wp.transform_get_rotation(ee_init_tf)
    tcp_pos_offset = wp.vec3(0.0, 0.0, wp.static(_ROBOTIQ_TCP_OFFSET_M))
    tcp_pos_prev = wp.transform_point(ee_init_tf, tcp_pos_offset)

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
    tcp_pos_offset = wp.vec3(0.0, 0.0, wp.static(_ROBOTIQ_TCP_OFFSET_M))
    ee_pos_actual[tid] = wp.transform_point(body_q[body_idx], tcp_pos_offset)


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


# ---- Brick mesh generation ----


def _brick_cylinder_mesh(radius, height, segments, cx=0.0, cy=0.0, cz=0.0, bottom_cap=True):
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


def _brick_combine_meshes(mesh_list):
    all_v, all_f, off = [], [], 0
    for v, f in mesh_list:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v).astype(np.float32), np.vstack(all_f).astype(np.int32)


def _make_brick_shell_mesh(nx, ny):
    ox = nx * _BRICK_PITCH / 2.0
    oy = ny * _BRICK_PITCH / 2.0
    inx = ox - _BRICK_WALL_THICKNESS
    iny = oy - _BRICK_WALL_THICKNESS
    H = _BRICK_HEIGHT
    T = _BRICK_TOP_THICKNESS
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


def _make_brick_mesh(nx=4, ny=2):
    shell_v, shell_f = _make_brick_shell_mesh(nx, ny)
    seg = _BRICK_CYLINDER_SEGMENTS
    stud_meshes = []
    for i in range(nx):
        for j in range(ny):
            sx = (i - (nx - 1) / 2.0) * _BRICK_PITCH
            sy = (j - (ny - 1) / 2.0) * _BRICK_PITCH
            stud_meshes.append(
                _brick_cylinder_mesh(
                    _BRICK_STUD_RADIUS,
                    _BRICK_STUD_HEIGHT,
                    seg,
                    cx=sx,
                    cy=sy,
                    cz=_BRICK_HEIGHT,
                    bottom_cap=False,
                )
            )
    tube_meshes = []
    if ny == 2:
        tube_height = _BRICK_HEIGHT - _BRICK_TOP_THICKNESS
        for i in range(nx - 1):
            tx = (i - (nx - 2) / 2.0) * _BRICK_PITCH
            tube_meshes.append(_brick_cylinder_mesh(_BRICK_TUBE_OUTER_RADIUS, tube_height, seg, cx=tx, cy=0.0, cz=0.0))
    v, f = _brick_combine_meshes([(shell_v, shell_f), *stud_meshes, *tube_meshes])
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
