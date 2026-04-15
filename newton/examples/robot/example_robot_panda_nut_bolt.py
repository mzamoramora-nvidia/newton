# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Panda Nut Bolt
#
# Demonstrates a Franka Panda robot picking up an M20 nut and placing it
# on a bolt using hydroelastic contacts, gravity compensation, and
# IK-based control. The nut threads onto the bolt under gravity after
# release.
#
# Command: python -m newton.examples robot_panda_nut_bolt --world-count 4
#
###########################################################################

import copy
import enum
from dataclasses import replace

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact

# IsaacGymEnvs nut/bolt assets
ASSEMBLY_STR = "m20_loose"
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

# SDF parameters for nut/bolt meshes (high resolution for threaded geometry)
SDF_MAX_RESOLUTION_NUT_BOLT = 256
SDF_NARROW_BAND_NUT_BOLT = (-0.005, 0.005)

# SDF parameters for gripper/table meshes
SDF_MAX_RESOLUTION_GRIPPER = 64
SDF_NARROW_BAND_GRIPPER = (-0.01, 0.01)


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    STABILIZE = 3
    LIFT = 4
    MOVE_TO_BOLT = 5
    REFINE_PLACE = 6
    RELEASE = 7
    RETRACT = 8
    HOME = 9


def load_mesh_with_sdf(
    mesh_file: str,
    gap: float = 0.005,
    scale: float = 1.0,
    max_resolution: int = 256,
    narrow_band_range: tuple[float, float] = (-0.005, 0.005),
) -> tuple[newton.Mesh, wp.vec3, np.ndarray]:
    """Load a triangle mesh, center it, and build an SDF.

    Args:
        mesh_file: Mesh file path.
        gap: Contact margin [m].
        scale: Uniform mesh scale [unitless].
        max_resolution: SDF grid resolution.
        narrow_band_range: SDF narrow band range [m].

    Returns:
        Tuple of ``(mesh, center_vec, half_extents)`` where ``center_vec``
        is the recenter offset [m] and ``half_extents`` is the mesh
        half-extents [m] after scaling.
    """
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)

    min_extent = vertices.min(axis=0)
    max_extent = vertices.max(axis=0)
    center = (min_extent + max_extent) / 2
    half_extents = (max_extent - min_extent) / 2 * scale

    vertices = vertices - center
    center_vec = wp.vec3(center) * float(scale)

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=max_resolution,
        narrow_band_range=narrow_band_range,
        margin=gap,
        scale=(scale, scale, scale),
    )
    return mesh, center_vec, half_extents


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh: newton.Mesh,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    label: str | None = None,
    center_vec: wp.vec3 | None = None,
    scale: float = 1.0,
    floating: bool = True,
) -> int:
    """Add a mesh shape, optionally as a new floating body.

    Args:
        builder: Model builder.
        mesh: Mesh geometry with SDF data.
        transform: Body/shape transform with position [m] and orientation.
        shape_cfg: Shape configuration.
        label: Body/shape label.
        center_vec: Mesh center offset [m] (from :func:`load_mesh_with_sdf`).
        scale: Uniform mesh scale [unitless].
        floating: If ``True`` create a new dynamic body, otherwise attach
            to the world body (``body=-1``).

    Returns:
        Created body index, or ``-1`` if fixed.
    """
    if center_vec is not None:
        center_world = wp.quat_rotate(transform.q, center_vec)
        transform = wp.transform(transform.p + center_world, transform.q)

    if floating:
        body = builder.add_body(label=label, xform=transform)
        builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
        return body
    else:
        builder.add_shape_mesh(
            body=-1, mesh=mesh, scale=(scale, scale, scale), xform=transform, cfg=shape_cfg, label=label
        )
        return -1


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_soft_limits: wp.array[float],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    task_offset_approach: wp.vec3,
    task_offset_lift: wp.vec3,
    task_offset_bolt_approach: wp.vec3,
    task_offset_place: wp.vec3,
    task_offset_retract: wp.vec3,
    grasping_z_offset: float,
    grasp_yaw_offset: float,
    gripper_open_pos: float,
    gripper_closed_pos: float,
    bolt_place_pos: wp.vec3,
    home_pos: wp.vec3,
    task_init_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: int,
    nut_body_index: int,
    num_bodies_per_world: int,
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_target_interpolated: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_target_interpolated: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    tid = wp.tid()

    idx = task_idx[tid]
    task = task_schedule[idx]
    task_time_soft_limit = task_time_soft_limits[idx]

    task_time_elapsed[tid] += task_dt

    # Interpolation parameter t between 0 and 1
    t = wp.min(1.0, task_time_elapsed[tid] / task_time_soft_limit)

    # Get the EE position and rotation at the start of this task
    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_id])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_id])
    ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    # Get the current nut position and rotation
    nut_body_id = tid * num_bodies_per_world + nut_body_index
    nut_pos = wp.transform_get_translation(body_q[nut_body_id])
    nut_quat = wp.transform_get_rotation(body_q[nut_body_id])

    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        # Move above the nut, align gripper offset by grasp_yaw_offset about Z
        ee_pos_target[tid] = nut_pos + task_offset_approach
        yaw_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), grasp_yaw_offset)
        ee_quat_target = yaw_rot * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi) * wp.quat_inverse(nut_quat)
    elif task == TaskType.REFINE_APPROACH.value:
        # Descend to nut grasping height
        ee_pos_target[tid] = nut_pos + wp.vec3(0.0, 0.0, grasping_z_offset)
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        # Close gripper around nut
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.STABILIZE.value:
        # Hold position with gripper closed to let contact forces settle
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.LIFT.value:
        # Lift nut upward
        ee_pos_target[tid] = ee_pos_prev + task_offset_lift
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.MOVE_TO_BOLT.value:
        # Move above the bolt
        ee_pos_target[tid] = bolt_place_pos + task_offset_bolt_approach
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
        t_gripper = 1.0
    elif task == TaskType.REFINE_PLACE.value:
        # Lower nut onto bolt top
        ee_pos_target[tid] = bolt_place_pos + task_offset_place
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        # Open gripper to release nut onto bolt threads
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0 - t
    elif task == TaskType.RETRACT.value:
        # Move upward away from bolt
        ee_pos_target[tid] = ee_pos_prev + task_offset_retract
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
    elif task == TaskType.HOME.value:
        ee_pos_target[tid] = home_pos
    else:
        ee_pos_target[tid] = home_pos

    ee_pos_target_interpolated[tid] = ee_pos_prev * (1.0 - t) + ee_pos_target[tid] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[tid] = ee_quat_target[:4]
    ee_rot_target_interpolated[tid] = ee_quat_interpolated[:4]

    # Interpolate gripper between open and closed positions
    gripper_pos = gripper_open_pos * (1.0 - t_gripper) + gripper_closed_pos * t_gripper
    gripper_target[tid, 0] = gripper_pos
    gripper_target[tid, 1] = gripper_pos


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_soft_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    num_bodies_per_world: int,
    ee_index: int,
    pos_threshold_xy: float,
    pos_threshold_z: float,
    rot_threshold_deg: float,
    # outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_init_body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = task_idx[tid]
    task_time_soft_limit = task_time_soft_limits[idx]

    # Get the current position of the end-effector
    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_current = wp.transform_get_translation(body_q[ee_body_id])
    ee_quat_current = wp.transform_get_rotation(body_q[ee_body_id])

    # Per-axis position error
    pos_diff = ee_pos_target[tid] - ee_pos_current
    err_x = wp.abs(pos_diff[0])
    err_y = wp.abs(pos_diff[1])
    err_z = wp.abs(pos_diff[2])
    pos_settled = err_x < pos_threshold_xy and err_y < pos_threshold_xy and err_z < pos_threshold_z

    # Rotation error
    ee_quat_target = wp.quaternion(ee_rot_target[tid][:3], ee_rot_target[tid][3])
    quat_rel = ee_quat_current * wp.quat_inverse(ee_quat_target)
    rot_err = wp.abs(wp.degrees(2.0 * wp.atan2(wp.length(quat_rel[:3]), quat_rel[3])))
    rot_settled = rot_err < rot_threshold_deg

    # Advance when time elapsed, position settled, rotation settled, and not last task
    if (
        task_time_elapsed[tid] >= task_time_soft_limit
        and pos_settled
        and rot_settled
        and task_idx[tid] < wp.len(task_time_soft_limits) - 1
    ):
        task_idx[tid] += 1
        task_time_elapsed[tid] = 0.0

        body_id_start = tid * num_bodies_per_world
        for i in range(num_bodies_per_world):
            body_id = body_id_start + i
            task_init_body_q[body_id] = body_q[body_id]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.collide_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.test_mode = args.test
        self.viewer = viewer

        # Scene layout
        self.table_height = 0.1
        self.table_pos = wp.vec3(0.0, -0.5, 0.5 * self.table_height)
        self.table_top_center = self.table_pos + wp.vec3(0.0, 0.0, 0.5 * self.table_height)
        self.robot_base_pos = self.table_top_center + wp.vec3(-0.5, 0.0, 0.0)

        # Bolt and nut positions on the table
        self.bolt_base_pos = self.table_top_center + wp.vec3(0.1, 0.0, 0.0)
        self.nut_start_pos = self.table_top_center + wp.vec3(0.05, 0.15, 0.0)

        # Task offsets
        self.task_offset_approach = wp.vec3(0.0, 0.0, 0.04)
        self.task_offset_lift = wp.vec3(0.0, 0.0, 0.15)
        self.task_offset_bolt_approach = wp.vec3(0.0, 0.0, 0.06)
        self.task_offset_place = wp.vec3(0.0, 0.0, 0.001)
        self.task_offset_retract = wp.vec3(0.0, 0.0, 0.10)
        self.grasping_z_offset = 0.001
        self.grasp_yaw_offset = 30.0 * wp.pi / 180.0  # rotate gripper 30 deg about Z wrt nut

        # Per-axis advance thresholds
        self.pos_threshold_xy = 0.0005  # 0.5 mm
        self.pos_threshold_z = 0.00075  # 0.75 mm (tighter for vertical precision)
        self.rot_threshold_deg = 0.5  # degrees

        # Gripper open/closed positions [m per finger]
        # Panda gripper: each finger travels 0 (closed) to 0.04m (open), we use 0.06 as open target
        self.gripper_open_pos = 0.06
        # Grasp margin: close to nut half-width minus a margin to avoid over-squeezing.
        # For a regular hex nut, the grasp width depends on the angle relative to the flats.
        # width(theta) = across_flats / cos(theta_eff), where theta_eff is the angle from
        # the nearest flat normal (0 to 30 deg range due to 6-fold symmetry).
        nut_across_flats = 0.030  # M20 nut, 30mm
        # theta_eff: angle from the nearest flat normal (0-30 deg, 6-fold symmetry)
        rem = self.grasp_yaw_offset % (np.pi / 3)  # remainder in [0, 60 deg)
        theta_eff = rem if rem <= np.pi / 6 else np.pi / 3 - rem
        nut_grasp_width = nut_across_flats / np.cos(theta_eff)
        grasp_margin = 0.007  # 7mm extra closure past first contact
        self.gripper_closed_pos = max(0.0, nut_grasp_width / 2.0 - grasp_margin)
        gripper_ke = 100.0  # from joint_target_ke for finger joints
        expected_force_per_finger = gripper_ke * grasp_margin
        print(
            f"Grasp: yaw={np.degrees(self.grasp_yaw_offset):.0f}deg, "
            f"nut width={nut_grasp_width * 1000:.1f}mm, "
            f"margin={grasp_margin * 1000:.1f}mm, "
            f"finger target={self.gripper_closed_pos * 1000:.1f}mm, "
            f"expected force/finger={expected_force_per_finger:.3f}N"
        )

        # Download nut/bolt assets
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")

        # Load nut/bolt meshes with SDF
        bolt_mesh, bolt_center, bolt_half_extents = load_mesh_with_sdf(
            bolt_file,
            gap=0.005,
            max_resolution=SDF_MAX_RESOLUTION_NUT_BOLT,
            narrow_band_range=SDF_NARROW_BAND_NUT_BOLT,
        )
        nut_mesh, nut_center, _nut_half_extents = load_mesh_with_sdf(
            nut_file,
            gap=0.005,
            max_resolution=SDF_MAX_RESOLUTION_NUT_BOLT,
            narrow_band_range=SDF_NARROW_BAND_NUT_BOLT,
        )

        # Compute placement position above bolt top
        bolt_effective_center = wp.vec3(
            self.bolt_base_pos[0] + float(bolt_center[0]),
            self.bolt_base_pos[1] + float(bolt_center[1]),
            self.bolt_base_pos[2] + float(bolt_center[2]),
        )
        self.bolt_place_pos = wp.vec3(
            bolt_effective_center[0],
            bolt_effective_center[1],
            bolt_effective_center[2] + float(bolt_half_extents[2]) + 0.005,
        )

        # Store bolt frame for debug visualization (static, identity rotation)
        self.bolt_frame_pos = bolt_effective_center

        # Build robot with table, gravity compensation, and hydroelastic fingers
        robot_builder = self.build_franka_with_table()
        self.robot_body_count = robot_builder.body_count

        # Model for IK (robot only, no nut/bolt)
        self.model_single = copy.deepcopy(robot_builder).finalize()

        # Add bolt (fixed to ground) and nut (floating body)
        nut_bolt_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            mu=0.01,
            ke=1e7,
            kd=1e4,
            gap=0.005,
            density=8000.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
            is_hydroelastic=True,
        )

        bolt_xform = wp.transform(self.bolt_base_pos, wp.quat_identity())
        add_mesh_object(
            robot_builder,
            bolt_mesh,
            bolt_xform,
            nut_bolt_cfg,
            label="bolt",
            center_vec=bolt_center,
            floating=False,
        )

        self.nut_body_index = robot_builder.body_count
        nut_xform = wp.transform(
            self.nut_start_pos,
            # wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 8),
        )
        add_mesh_object(
            robot_builder,
            nut_mesh,
            nut_xform,
            nut_bolt_cfg,
            label="nut",
            center_vec=nut_center,
            floating=True,
        )

        # Build multi-world scene
        scene = newton.ModelBuilder()
        scene.replicate(robot_builder, self.world_count)
        scene.add_ground_plane()

        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count

        # Contact sensor: must be created BEFORE Contacts so the "force" attribute is requested.
        # Sense forces on nut bodies from finger shapes.
        self.contact_sensor = SensorContact(
            self.model,
            sensing_obj_bodies="nut",
            counterpart_bodies="*finger*",
        )

        # Collision pipeline with hydroelastic SDF
        self.rigid_contact_max = 1000 * self.world_count
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            output_contact_surface=hasattr(viewer, "renderer"),
            buffer_mult_iso=2,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=self.rigid_contact_max,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        self.contacts = self.collision_pipeline.contacts()

        # MuJoCo solver with Newton contacts
        num_contacts_per_world = self.rigid_contact_max // self.world_count
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=num_contacts_per_world,
            nconmax=num_contacts_per_world,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        # State and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        # IK and tasks
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.setup_ik()
        self.setup_tasks()

        # Viewer
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.show_isosurface = False
        self.show_debug_frames = True
        self.show_single_world = False

        # GUI state (updated periodically in step, displayed in render_ui)
        self._gui_selected_world = 0
        self._gui_task_name = "—"
        self._gui_task_timer = 0.0
        self._gui_pos_err = np.zeros(3)
        self._gui_rot_err = 0.0
        self._gui_frame = 0
        self._gui_read_interval = 4  # read GPU state every N frames
        self._gui_contact_count = 0

        # Per-world contact force tracking
        self._cur_force = np.zeros(self.world_count)  # current total force on nut [N]
        self._max_force = np.zeros(self.world_count)  # all-time max total force [N]
        self._cur_friction = np.zeros(self.world_count)  # current friction force [N]
        self._max_friction = np.zeros(self.world_count)  # all-time max friction force [N]
        # Per-counterpart (left/right finger) forces for selected world
        self._gui_finger_forces = np.zeros(2)  # [left, right] magnitudes [N]
        self._nut_z_at_lift_start = np.full(self.world_count, np.nan)
        self._ee_z_at_lift_start = np.full(self.world_count, np.nan)
        self._nut_z_at_lift_end = np.full(self.world_count, np.nan)
        self._ee_z_at_lift_end = np.full(self.world_count, np.nan)
        self._slip = np.full(self.world_count, np.nan)  # computed after lift [m]

        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)
            self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
            self.viewer.show_hydro_contact_surface = self.show_isosurface
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.capture()

    def build_franka_with_table(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            gap=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        shape_cfg_meshes = replace(shape_cfg, is_hydroelastic=True)

        builder.default_shape_cfg = shape_cfg

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )

        # Initial joint configuration
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[:9] = [*init_q, 0.05, 0.05]
        builder.joint_target_pos[:9] = [*init_q, 1.0, 1.0]

        # Joint gains (high values needed with gravity compensation)
        builder.joint_target_ke[:9] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        builder.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        # Gravity compensation on arm joints
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(7):
            gravcomp_attr.values[dof_idx] = True

        # Gravity compensation on arm and hand bodies
        # Body 0 = base, 1 = fr3_link0, 2-8 = fr3_link1-7,
        # 9-11 = fr3_link8/fr3_hand/fr3_hand_tcp, 12-13 = fingers
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0

        # Find finger and hand bodies for hydroelastic SDF
        def find_body(name):
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        finger_body_indices = {
            find_body("fr3_leftfinger"),
            find_body("fr3_rightfinger"),
            find_body("fr3_hand"),
        }
        self.ee_index = find_body("fr3_hand_tcp")

        # Enable hydroelastic SDF on finger/hand mesh shapes
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx in finger_body_indices and builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        builder.shape_source[shape_idx] = mesh
                        builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=SDF_MAX_RESOLUTION_GRIPPER,
                        narrow_band_range=SDF_NARROW_BAND_GRIPPER,
                        margin=shape_cfg.gap,
                    )
                builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
            elif body_idx not in finger_body_indices:
                non_finger_shape_indices.append(shape_idx)

        # Convert non-finger shapes to convex hulls
        builder.approximate_meshes(
            method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        )

        # Table (hydroelastic mesh on world body)
        table_mesh = newton.Mesh.create_box(
            0.4,
            0.4,
            0.5 * self.table_height,
            duplicate_vertices=True,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=True,
        )
        table_mesh.build_sdf(
            max_resolution=SDF_MAX_RESOLUTION_GRIPPER,
            narrow_band_range=SDF_NARROW_BAND_GRIPPER,
            margin=shape_cfg.gap,
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=shape_cfg_meshes,
        )

        return builder

    def setup_ik(self):
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        init_ee_pos = body_q_np[self.ee_index][:3]
        self.home_pos = wp.vec3(init_ee_pos)

        # Position objective (per-world)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([self.home_pos] * self.world_count, dtype=wp.vec3),
        )

        # Rotation objective (per-world)
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.transform_get_rotation(self.ee_tf)[:4]] * self.world_count, dtype=wp.vec4),
        )

        ik_dofs = self.model_single.joint_coord_count

        # Joint limit objective
        self.joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, :ik_dofs])
        self.joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, :ik_dofs])
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.joint_limit_lower.flatten(),
            joint_limit_upper=self.joint_limit_upper.flatten(),
        )

        # Variables the solver will update
        self.joint_q_ik = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, :ik_dofs])

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def setup_tasks(self):
        task_schedule = [
            TaskType.APPROACH,
            TaskType.REFINE_APPROACH,
            TaskType.GRASP,
            TaskType.STABILIZE,
            TaskType.LIFT,
            TaskType.MOVE_TO_BOLT,
            TaskType.REFINE_PLACE,
            TaskType.RELEASE,
            TaskType.RETRACT,
            TaskType.HOME,
        ]
        task_time_soft_limits = [1.5, 1.0, 0.5, 1.0, 1.5, 2.0, 1.5, 0.5, 1.0, 2.0]

        self.task_counter = len(task_schedule)
        self.task_schedule = wp.array(task_schedule, dtype=wp.int32)
        self.task_time_soft_limits = wp.array(task_time_soft_limits, dtype=float)

        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.task_idx = wp.zeros(self.world_count, dtype=wp.int32)
        self.task_dt = self.frame_dt
        self.task_time_elapsed = wp.zeros(self.world_count, dtype=wp.float32)

        # Target arrays
        self.ee_pos_target = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_pos_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(self.world_count, dtype=wp.vec4)
        self.ee_rot_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec4)
        self.gripper_target = wp.zeros(shape=(self.world_count, 2), dtype=wp.float32)

    def set_joint_targets(self):
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_schedule,
                self.task_time_soft_limits,
                self.task_idx,
                self.task_time_elapsed,
                self.task_dt,
                self.task_offset_approach,
                self.task_offset_lift,
                self.task_offset_bolt_approach,
                self.task_offset_place,
                self.task_offset_retract,
                self.grasping_z_offset,
                self.grasp_yaw_offset,
                self.gripper_open_pos,
                self.gripper_closed_pos,
                self.bolt_place_pos,
                self.home_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
                self.nut_body_index,
                self.num_bodies_per_world,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_target_interpolated,
                self.ee_rot_target,
                self.ee_rot_target_interpolated,
                self.gripper_target,
            ],
        )

        # Set IK targets
        self.pos_obj.set_target_positions(self.ee_pos_target_interpolated)
        self.rot_obj.set_target_rotations(self.ee_rot_target_interpolated)

        # Solve IK
        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # Set joint target positions from IK solution + gripper targets
        joint_target_view = self.control.joint_target_pos.reshape((self.world_count, -1))
        wp.copy(dest=joint_target_view[:, :7], src=self.joint_q_ik[:, :7])
        wp.copy(dest=joint_target_view[:, 7:9], src=self.gripper_target[:, :2])

        # Advance tasks when conditions are met
        wp.launch(
            advance_task_kernel,
            dim=self.world_count,
            inputs=[
                self.task_time_soft_limits,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.num_bodies_per_world,
                self.ee_index,
                self.pos_threshold_xy,
                self.pos_threshold_z,
                self.rot_threshold_deg,
            ],
            outputs=[
                self.task_idx,
                self.task_time_elapsed,
                self.task_init_body_q,
            ],
        )

    def capture(self):
        self.capture_sim()
        self.capture_ik()

    def capture_sim(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def capture_ik(self):
        self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.set_joint_targets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self._gui_frame += 1

        # Periodic readback (avoid every-frame GPU sync)
        if self._gui_frame % self._gui_read_interval == 0:
            self._update_gui_state()
            self._update_contact_and_slip()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_hydro_contact_surface(
            (
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                if self.collision_pipeline.hydroelastic_sdf is not None
                else None
            ),
            penetrating_only=True,
        )
        if self.show_debug_frames:
            self._log_debug_frames()
        else:
            self.viewer.log_lines("/debug_frames", None, None, None)
        self.viewer.end_frame()

    def _log_debug_frames(self):
        """Draw bolt, EE IK target, and nut frames as RGB axis lines per world."""
        axis_len = 0.05
        axes = [
            (wp.vec3(1.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)),  # X axis, red
            (wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 1.0, 0.0)),  # Y axis, green
            (wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 1.0)),  # Z axis, blue
        ]

        starts = []
        ends = []
        colors = []

        def add_frame(pos, q):
            for axis_dir, color in axes:
                starts.append(pos)
                ends.append(pos + wp.quat_rotate(q, axis_dir) * axis_len)
                colors.append(color)

        # Read world offsets
        offsets = self.viewer.world_offsets.numpy() if self.viewer.world_offsets is not None else None

        # Read EE targets and body transforms (GPU -> CPU)
        ee_pos_np = self.ee_pos_target_interpolated.numpy()
        ee_rot_np = self.ee_rot_target_interpolated.numpy()
        body_q_np = self.state_0.body_q.numpy()

        for w in range(self.world_count):
            off = (
                wp.vec3(float(offsets[w][0]), float(offsets[w][1]), float(offsets[w][2]))
                if offsets is not None
                else wp.vec3()
            )

            # Bolt frame (static, identity rotation)
            add_frame(self.bolt_frame_pos + off, wp.quat_identity())

            # EE IK target frame
            e = ee_pos_np[w]
            eq = ee_rot_np[w]
            add_frame(
                wp.vec3(float(e[0]), float(e[1]), float(e[2])) + off,
                wp.quat(float(eq[0]), float(eq[1]), float(eq[2]), float(eq[3])),
            )

            # Nut frame (dynamic body)
            nq = body_q_np[w * self.num_bodies_per_world + self.nut_body_index]
            add_frame(
                wp.vec3(float(nq[0]), float(nq[1]), float(nq[2])) + off,
                wp.quat(float(nq[3]), float(nq[4]), float(nq[5]), float(nq[6])),
            )

        self.viewer.log_lines(
            "/debug_frames",
            wp.array(starts, dtype=wp.vec3),
            wp.array(ends, dtype=wp.vec3),
            wp.array(colors, dtype=wp.vec3),
        )

    def _update_gui_state(self):
        """Read GPU state for the selected world (called periodically from step)."""
        w = self._gui_selected_world

        # Current task name and timer
        task_val = int(self.task_idx.numpy()[w])
        self._gui_task_name = TaskType(self.task_schedule.numpy()[task_val]).name
        self._gui_task_timer = float(self.task_time_elapsed.numpy()[w])

        # EE position error (target vs actual)
        body_q = self.state_0.body_q.numpy()
        ee_body_id = w * self.num_bodies_per_world + self.ee_index
        ee_pos = body_q[ee_body_id][:3]
        ee_target = self.ee_pos_target.numpy()[w]
        self._gui_pos_err = ee_target - ee_pos

        # EE rotation error (degrees)
        ee_quat = body_q[ee_body_id][3:7]
        target_vec4 = self.ee_rot_target.numpy()[w]
        # q_rel = q_current * inv(q_target)
        q_cur = ee_quat
        q_tgt = target_vec4
        dot = abs(np.dot(q_cur, q_tgt))
        dot = min(dot, 1.0)
        self._gui_rot_err = np.degrees(2.0 * np.arccos(dot))

    def _update_contact_and_slip(self):
        """Track contact forces and vertical slippage (called periodically from step)."""
        # Update contact forces via the solver and sensor
        self._gui_contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        self.solver.update_contacts(self.contacts, self.state_0)
        self.contact_sensor.update(self.state_0, self.contacts)

        # Read per-world total force and friction on the nut from finger contacts
        total_force_np = self.contact_sensor.total_force.numpy()  # (world_count, 3)
        friction_force_np = self.contact_sensor.total_force_friction.numpy()  # (world_count, 3)
        for w in range(self.world_count):
            f_mag = float(np.linalg.norm(total_force_np[w]))
            fr_mag = float(np.linalg.norm(friction_force_np[w]))
            self._cur_force[w] = f_mag
            self._cur_friction[w] = fr_mag
            self._max_force[w] = max(self._max_force[w], f_mag)
            self._max_friction[w] = max(self._max_friction[w], fr_mag)

        # Read per-finger forces for the selected world (from force_matrix)
        w = self._gui_selected_world
        if self.contact_sensor.force_matrix is not None:
            fm = self.contact_sensor.force_matrix.numpy()  # (world_count, n_counterparts, 3)
            n_cols = fm.shape[1]
            for c in range(min(n_cols, 2)):
                self._gui_finger_forces[c] = float(np.linalg.norm(fm[w, c]))

        # Slippage: snapshot Z positions at LIFT start and MOVE_TO_BOLT start
        tasks_np = self.task_idx.numpy()
        schedule_np = self.task_schedule.numpy()
        body_q = self.state_0.body_q.numpy()

        for w in range(self.world_count):
            task_type = int(schedule_np[int(tasks_np[w])])
            ee_body_id = w * self.num_bodies_per_world + self.ee_index
            nut_body_id = w * self.num_bodies_per_world + self.nut_body_index
            ee_z = float(body_q[ee_body_id][2])
            nut_z = float(body_q[nut_body_id][2])

            # Record at LIFT start (first frame we see LIFT)
            if task_type == TaskType.LIFT and np.isnan(self._nut_z_at_lift_start[w]):
                self._nut_z_at_lift_start[w] = nut_z
                self._ee_z_at_lift_start[w] = ee_z

            # Record at MOVE_TO_BOLT (lift finished)
            if task_type == TaskType.MOVE_TO_BOLT:
                self._nut_z_at_lift_end[w] = nut_z
                self._ee_z_at_lift_end[w] = ee_z
                # Compute slip: how much more the EE lifted than the nut
                if not np.isnan(self._nut_z_at_lift_start[w]):
                    ee_lift = ee_z - self._ee_z_at_lift_start[w]
                    nut_lift = nut_z - self._nut_z_at_lift_start[w]
                    self._slip[w] = ee_lift - nut_lift

    def render_ui(self, imgui):
        imgui.separator()

        # World selector
        changed, val = imgui.slider_int("World", self._gui_selected_world, 0, self.world_count - 1)
        if changed:
            self._gui_selected_world = val
            if self.show_single_world:
                self.viewer.set_visible_worlds([val])

        changed, self.show_single_world = imgui.checkbox("Show Single World", self.show_single_world)
        if changed:
            if self.show_single_world:
                self.viewer.set_visible_worlds([self._gui_selected_world])
            else:
                self.viewer.set_visible_worlds(None)

        imgui.separator()

        # Task state machine info
        imgui.text(f"Task: {self._gui_task_name}")
        imgui.text(f"Timer: {self._gui_task_timer:.2f}s")

        # Per-axis position error with threshold check
        err = self._gui_pos_err * 1000.0  # mm
        limit_xy = self.pos_threshold_xy * 1000.0
        limit_z = self.pos_threshold_z * 1000.0
        ok_x = "*" if abs(err[0]) < limit_xy else " "
        ok_y = "*" if abs(err[1]) < limit_xy else " "
        ok_z = "*" if abs(err[2]) < limit_z else " "
        imgui.text(f"Pos x:{err[0]:+.2f}mm {ok_x}  y:{err[1]:+.2f}mm {ok_y}  z:{err[2]:+.2f}mm {ok_z}")
        imgui.text(f"  limit: xy<{limit_xy:.2f}mm  z<{limit_z:.2f}mm")

        # Rotation error with threshold check
        ok_r = "*" if self._gui_rot_err < self.rot_threshold_deg else " "
        imgui.text(f"Rot err: {self._gui_rot_err:.2f} deg {ok_r}  limit<{self.rot_threshold_deg:.1f}")

        imgui.text(f"Frame: {self._gui_frame}  Sim: {self.sim_time:.2f}s")

        imgui.separator()

        # Contact forces for selected world
        w = self._gui_selected_world
        imgui.text(f"Contacts: {self._gui_contact_count}")
        imgui.text(f"Total force:    {self._cur_force[w]:.3f} N  (max: {self._max_force[w]:.3f} N)")
        imgui.text(f"Friction force: {self._cur_friction[w]:.3f} N  (max: {self._max_friction[w]:.3f} N)")
        imgui.text(f"Finger L: {self._gui_finger_forces[0]:.3f} N  R: {self._gui_finger_forces[1]:.3f} N")

        # Slippage for selected world
        if not np.isnan(self._slip[w]):
            imgui.text(f"Slip: {self._slip[w] * 1000:.2f} mm")
        else:
            imgui.text("Slip: —")

        imgui.separator()

        # Visualization toggles
        changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_isosurface
        _, self.show_debug_frames = imgui.checkbox("Show Debug Frames", self.show_debug_frames)

    def test_final(self):
        body_q = self.state_0.body_q.numpy()

        bolt_xy = np.array([float(self.bolt_place_pos[0]), float(self.bolt_place_pos[1])])
        table_top_z = float(self.table_top_center[2])

        # Print stats for all worlds first (before assertions)
        print("\nPer-world report:")
        errors = []
        for world_id in range(self.world_count):
            nut_body_id = world_id * self.num_bodies_per_world + self.nut_body_index
            nut_pos = body_q[nut_body_id][:3]
            nut_xy = nut_pos[:2]
            xy_error = float(np.linalg.norm(nut_xy - bolt_xy))

            f_max = self._max_force[world_id]
            fr_max = self._max_friction[world_id]
            slip = self._slip[world_id]
            slip_str = f"{slip * 1000:.2f}mm" if not np.isnan(slip) else "n/a"
            lift_str = ""
            if not np.isnan(slip):
                nut_lift = self._nut_z_at_lift_end[world_id] - self._nut_z_at_lift_start[world_id]
                lift_str = f"  lift={nut_lift * 1000:.1f}mm"

            print(
                f"  World {world_id}: "
                f"nut=({nut_pos[0]:.4f}, {nut_pos[1]:.4f}, {nut_pos[2]:.4f})  "
                f"xy_err={xy_error * 1000:.1f}mm  "
                f"force_max={f_max:.3f}N  friction_max={fr_max:.3f}N  "
                f"slip={slip_str}{lift_str}"
            )

            # Collect errors instead of asserting immediately
            if not np.all(np.isfinite(nut_pos)):
                errors.append(f"World {world_id}: Nut has non-finite position {nut_pos}")
            elif xy_error >= 0.02:
                errors.append(f"World {world_id}: Nut XY too far from bolt. Error={xy_error:.4f}m (max 0.02m)")
            elif nut_pos[2] <= table_top_z:
                errors.append(f"World {world_id}: Nut below table. z={nut_pos[2]:.4f}")
            elif nut_pos[2] >= table_top_z + 0.06:
                errors.append(f"World {world_id}: Nut too high. z={nut_pos[2]:.4f}")

        if errors:
            raise AssertionError("\n".join(errors))

        print(f"test_final passed for all {self.world_count} worlds")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=1200)
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
