"""Build student-facing robot sensor snapshots from simulation state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import atan2, degrees, hypot, isinf, radians
from typing import Any, cast

from racing.student.api import (
    DEFAULT_CAMERA_LOOKAHEAD_DISTANCES_M,
    DEFAULT_LIDAR_ANGLES_DEGREES,
    DEFAULT_LIDAR_MAX_DISTANCE_M,
    MAX_CAMERA_COMPETITORS,
    CameraCompetitorReading,
    CameraSensors,
    ContactSensors,
    ImuSensors,
    LidarSensors,
    OdometrySensors,
    RobotSensors,
)
from racing.physics import RobotVehicle
from racing.race.progress import (
    TrackProgressModel,
    heading_error_degrees,
    project_track_position,
    track_pose_at_distance,
)
from racing.track.spatial import node_position, track_forward_vector
from racing.track.world import TRACK_WIDTH, TrackPoint, clamp

LIDAR_SENSOR_HEIGHT_M = 0.32
LIDAR_START_OFFSET_M = 0.62
LIDAR_INFINITE_RAYCAST_DISTANCE_M = 1_000.0
CAMERA_TRACK_WIDTH_VIEW_M = TRACK_WIDTH * 0.65
CAMERA_MAX_HEADING_ERROR_DEGREES = 90.0
CAMERA_COMPETITOR_MIN_DISTANCE_M = 0.05
IGNORED_LIDAR_NODE_NAMES: frozenset[str] = frozenset({"grass-and-track-floor"})
EMPTY_NODE_NAMES: frozenset[str] = frozenset()
EMPTY_ROBOTS: tuple[RobotVehicle, ...] = ()


@dataclass(frozen=True, slots=True)
class RobotSensorBuilderState:
    """Previous values needed to derive rates and odometry for the next tick."""

    time_s: float = 0.0
    position: TrackPoint | None = None
    heading_degrees: float = 0.0
    speed_mps: float = 0.0
    distance_m: float = 0.0
    wall_contact_s: float = 0.0
    robot_contact_s: float = 0.0
    tick: int = -1


def build_robot_sensors(
    *,
    physics_world: Any,
    robot: RobotVehicle,
    track_model: TrackProgressModel,
    time_s: float,
    dt_s: float,
    previous_state: RobotSensorBuilderState | None = None,
    other_robot_node_names: frozenset[str] = EMPTY_NODE_NAMES,
    other_robots: tuple[RobotVehicle, ...] = EMPTY_ROBOTS,
) -> tuple[RobotSensors, RobotSensorBuilderState]:
    """Build the sensor object passed into a student controller."""
    state = RobotSensorBuilderState() if previous_state is None else previous_state
    tick = state.tick + 1
    bounded_dt_s = max(0.0, dt_s)
    x_m, _y_m, z_m = node_position(robot.chassis_np)
    position = TrackPoint(x=x_m, z=z_m)
    heading_degrees = _node_heading_degrees(robot.chassis_np)
    pitch_degrees = _node_pitch_degrees(robot.chassis_np)
    roll_degrees = _node_roll_degrees(robot.chassis_np)
    speed_mps = _robot_speed_mps(robot)
    yaw_rate_degrees_per_s = _rate_per_second(
        heading_error_degrees(
            current_heading_degrees=state.heading_degrees,
            target_heading_degrees=heading_degrees,
        ),
        bounded_dt_s,
        enabled=state.tick >= 0,
    )
    forward_acceleration_mps2 = _rate_per_second(speed_mps - state.speed_mps, bounded_dt_s, enabled=state.tick >= 0)
    lateral_acceleration_mps2 = speed_mps * radians(yaw_rate_degrees_per_s)
    distance_m = state.distance_m + abs(speed_mps) * bounded_dt_s
    chassis_name = _node_name(robot.chassis_np.node()) if hasattr(robot.chassis_np, "node") else ""
    ignored_node_names = frozenset((*IGNORED_LIDAR_NODE_NAMES, chassis_name))
    camera_competitor_robots = tuple(
        other_robot
        for other_robot in other_robots
        if other_robot is not robot and not bool(getattr(other_robot, "eliminated", False))
    )

    instantaneous_contact = contact_sensors(
        physics_world=physics_world,
        chassis_np=robot.chassis_np,
        other_robot_node_names=other_robot_node_names,
        damage=float(getattr(robot, "damage", 0.0)),
    )
    wall_contact_s = _continuous_contact_seconds(
        previous_seconds=state.wall_contact_s,
        active=instantaneous_contact.wall > 0.0,
        dt_s=bounded_dt_s,
    )
    robot_contact_s = _continuous_contact_seconds(
        previous_seconds=state.robot_contact_s,
        active=instantaneous_contact.robot > 0.0,
        dt_s=bounded_dt_s,
    )

    sensors = RobotSensors(
        dt_s=bounded_dt_s,
        tick=tick,
        imu=ImuSensors(
            heading_degrees=heading_degrees,
            yaw_rate_degrees_per_s=yaw_rate_degrees_per_s,
            pitch_degrees=pitch_degrees,
            roll_degrees=roll_degrees,
            forward_acceleration_mps2=forward_acceleration_mps2,
            lateral_acceleration_mps2=lateral_acceleration_mps2,
        ),
        odometry=OdometrySensors(
            speed_mps=speed_mps,
            distance_m=distance_m,
        ),
        lidar=lidar_sensors(
            physics_world=physics_world,
            chassis_np=robot.chassis_np,
            ignored_node_names=ignored_node_names,
        ),
        wall_lidar=lidar_sensors(
            physics_world=physics_world,
            chassis_np=robot.chassis_np,
            ignored_node_names=ignored_node_names,
            hit_filter=_node_name_is_track_barrier,
        ),
        camera=camera_sensors_from_track(
            model=track_model,
            position=position,
            heading_degrees=heading_degrees,
            competitors=camera_competitor_readings(
                physics_world=physics_world,
                position=(x_m, _y_m, z_m),
                heading_degrees=heading_degrees,
                speed_mps=speed_mps,
                competitor_chassis_nps=tuple(other_robot.chassis_np for other_robot in camera_competitor_robots),
                competitor_speeds_mps=tuple(_robot_speed_mps(other_robot) for other_robot in camera_competitor_robots),
            ),
        ),
        contact=ContactSensors(
            wall=wall_contact_s,
            robot=robot_contact_s,
            damage=instantaneous_contact.damage,
        ),
    )
    next_state = RobotSensorBuilderState(
        time_s=time_s,
        position=position,
        heading_degrees=heading_degrees,
        speed_mps=speed_mps,
        distance_m=distance_m,
        wall_contact_s=wall_contact_s,
        robot_contact_s=robot_contact_s,
        tick=tick,
    )
    return sensors, next_state


def lidar_sensors(
    *,
    physics_world: Any,
    chassis_np: Any,
    angles_degrees: tuple[float, ...] = DEFAULT_LIDAR_ANGLES_DEGREES,
    max_distance_m: float = DEFAULT_LIDAR_MAX_DISTANCE_M,
    ignored_node_names: frozenset[str] = IGNORED_LIDAR_NODE_NAMES,
    hit_filter: Callable[[str], bool] | None = None,
) -> LidarSensors:
    """Cast simple distance rays around the car."""
    if max_distance_m <= 0.0:
        raise ValueError("lidar max distance must be positive")

    core = cast(Any, import_module("panda3d.core"))
    x_m, y_m, z_m = node_position(chassis_np)
    heading_degrees = _node_heading_degrees(chassis_np)
    ray_y_m = y_m + LIDAR_SENSOR_HEIGHT_M
    distances_m: list[float] = []
    raycast_distance_m = LIDAR_INFINITE_RAYCAST_DISTANCE_M if isinf(max_distance_m) else max_distance_m

    for angle_degrees in angles_degrees:
        direction_x, direction_z = track_forward_vector(heading_degrees + angle_degrees)
        start = core.Vec3(
            x_m + direction_x * LIDAR_START_OFFSET_M,
            ray_y_m,
            z_m + direction_z * LIDAR_START_OFFSET_M,
        )
        end = core.Vec3(
            x_m + direction_x * (LIDAR_START_OFFSET_M + raycast_distance_m),
            ray_y_m,
            z_m + direction_z * (LIDAR_START_OFFSET_M + raycast_distance_m),
        )
        result = (
            physics_world.rayTestAll(start, end)
            if hit_filter is not None and hasattr(physics_world, "rayTestAll")
            else physics_world.rayTestClosest(start, end)
        )
        distances_m.append(
            _lidar_distance_from_ray_result(
                result,
                max_distance_m=max_distance_m,
                raycast_distance_m=raycast_distance_m,
                ignored_node_names=ignored_node_names,
                hit_filter=hit_filter,
            )
        )

    return LidarSensors(
        angles_degrees=angles_degrees,
        distances_m=tuple(distances_m),
        max_distance_m=max_distance_m,
    )


def camera_sensors_from_track(
    *,
    model: TrackProgressModel,
    position: TrackPoint,
    heading_degrees: float,
    lookahead_distances_m: tuple[float, ...] = DEFAULT_CAMERA_LOOKAHEAD_DISTANCES_M,
    competitors: tuple[CameraCompetitorReading, ...] = (),
) -> CameraSensors:
    """Report where the car is relative to the track centerline."""
    projection = project_track_position(model, position)
    center_offset_m = _lateral_offset_to_point_m(
        origin=position,
        heading_degrees=heading_degrees,
        target=projection.nearest_center,
    )
    desired_heading_error_degrees = heading_error_degrees(
        current_heading_degrees=heading_degrees,
        target_heading_degrees=projection.heading_degrees,
    )
    lookahead_offsets_m = tuple(
        _lateral_offset_to_point_m(
            origin=position,
            heading_degrees=heading_degrees,
            target=track_pose_at_distance(model, projection.progress_distance_m + lookahead_m).position,
        )
        for lookahead_m in lookahead_distances_m
    )
    return CameraSensors(
        visible=True,
        center_offset_m=center_offset_m,
        heading_error_degrees=desired_heading_error_degrees,
        lookahead_offsets_m=lookahead_offsets_m,
        lookahead_distances_m=lookahead_distances_m,
        competitors=competitors,
    )


def camera_competitor_readings(
    *,
    physics_world: Any | None = None,
    position: tuple[float, float, float],
    heading_degrees: float,
    speed_mps: float = 0.0,
    competitor_chassis_nps: tuple[Any, ...] = (),
    competitor_speeds_mps: tuple[float, ...] = (),
    max_competitors: int = MAX_CAMERA_COMPETITORS,
) -> tuple[CameraCompetitorReading, ...]:
    """Report visible opponent cars from nearest to farthest."""
    if max_competitors < 0:
        raise ValueError("max camera competitors cannot be negative")
    if max_competitors == 0 or not competitor_chassis_nps:
        return ()

    x_m, _y_m, z_m = position
    targets: list[CameraCompetitorReading] = []
    for index, competitor_chassis_np in enumerate(competitor_chassis_nps):
        competitor_x_m, _competitor_y_m, competitor_z_m = node_position(competitor_chassis_np)
        delta_x_m = competitor_x_m - x_m
        delta_z_m = competitor_z_m - z_m
        distance_m = hypot(delta_x_m, delta_z_m)
        if distance_m <= CAMERA_COMPETITOR_MIN_DISTANCE_M:
            continue
        if _camera_line_of_sight_blocked(
            physics_world=physics_world,
            start=position,
            end=(competitor_x_m, _competitor_y_m, competitor_z_m),
        ):
            continue
        competitor_speed_mps = _tuple_value(competitor_speeds_mps, index, default=0.0)
        targets.append(
            CameraCompetitorReading(
                distance_m,
                _relative_angle_to_delta_degrees(
                    delta_x_m=delta_x_m,
                    delta_z_m=delta_z_m,
                    heading_degrees=heading_degrees,
                ),
                relative_heading_degrees=heading_error_degrees(
                    current_heading_degrees=heading_degrees,
                    target_heading_degrees=_node_heading_degrees(competitor_chassis_np),
                ),
                speed_mps=competitor_speed_mps,
                closing_speed_mps=speed_mps - competitor_speed_mps,
            )
        )

    targets.sort(key=lambda target: target.distance_m)
    return tuple(targets[:max_competitors])


def contact_sensors(
    *,
    physics_world: Any,
    chassis_np: Any,
    other_robot_node_names: frozenset[str] = EMPTY_NODE_NAMES,
    damage: float = 0.0,
) -> ContactSensors:
    """Report whether the car is touching walls or other cars."""
    wall_contact = False
    robot_contact = False
    contact_result = physics_world.contactTest(chassis_np.node())
    for contact in contact_result.getContacts():
        node_names = (_node_name(contact.getNode0()), _node_name(contact.getNode1()))
        wall_contact = wall_contact or any(name.startswith("track-barrier") for name in node_names)
        robot_contact = robot_contact or any(
            name in other_robot_node_names or name.startswith("headless-blocker") for name in node_names
        )
    return ContactSensors(
        wall=1.0 if wall_contact else 0.0,
        robot=1.0 if robot_contact else 0.0,
        damage=damage,
    )


def _lidar_distance_from_ray_result(
    result: Any,
    *,
    max_distance_m: float,
    raycast_distance_m: float,
    ignored_node_names: frozenset[str],
    hit_filter: Callable[[str], bool] | None,
) -> float:
    if hasattr(result, "getHits"):
        return _lidar_distance_from_all_ray_result(
            result,
            max_distance_m=max_distance_m,
            raycast_distance_m=raycast_distance_m,
            ignored_node_names=ignored_node_names,
            hit_filter=hit_filter,
        )
    if not hasattr(result, "hasHit") or not bool(result.hasHit()):
        return max_distance_m
    hit_distance_m = _lidar_distance_from_hit(
        result,
        raycast_distance_m=raycast_distance_m,
        ignored_node_names=ignored_node_names,
        hit_filter=hit_filter,
    )
    return max_distance_m if hit_distance_m is None else hit_distance_m


def _lidar_distance_from_all_ray_result(
    result: Any,
    *,
    max_distance_m: float,
    raycast_distance_m: float,
    ignored_node_names: frozenset[str],
    hit_filter: Callable[[str], bool] | None,
) -> float:
    hits = result.getHits()
    best_distance_m = max_distance_m
    for hit in hits:
        hit_distance_m = _lidar_distance_from_hit(
            hit,
            raycast_distance_m=raycast_distance_m,
            ignored_node_names=ignored_node_names,
            hit_filter=hit_filter,
        )
        if hit_distance_m is not None and hit_distance_m < best_distance_m:
            best_distance_m = hit_distance_m
    return best_distance_m


def _lidar_distance_from_hit(
    hit: Any,
    *,
    raycast_distance_m: float,
    ignored_node_names: frozenset[str],
    hit_filter: Callable[[str], bool] | None,
) -> float | None:
    hit_node_name = _node_name(hit.getNode()) if hasattr(hit, "getNode") else ""
    if hit_node_name in ignored_node_names:
        return None
    if hit_filter is not None and not hit_filter(hit_node_name):
        return None
    hit_fraction = float(hit.getHitFraction()) if hasattr(hit, "getHitFraction") else 1.0
    return clamp(hit_fraction, 0.0, 1.0) * raycast_distance_m


def _node_name_is_track_barrier(node_name: str) -> bool:
    return node_name.startswith("track-barrier")


def _relative_angle_to_delta_degrees(
    *,
    delta_x_m: float,
    delta_z_m: float,
    heading_degrees: float,
) -> float:
    forward_x, forward_z = track_forward_vector(heading_degrees)
    right_x, right_z = forward_z, -forward_x
    forward_distance_m = delta_x_m * forward_x + delta_z_m * forward_z
    right_distance_m = delta_x_m * right_x + delta_z_m * right_z
    return degrees(atan2(right_distance_m, forward_distance_m))


def _lateral_offset_to_point_m(*, origin: TrackPoint, heading_degrees: float, target: TrackPoint) -> float:
    forward_x, forward_z = track_forward_vector(heading_degrees)
    right_x, right_z = forward_z, -forward_x
    return (target.x - origin.x) * right_x + (target.z - origin.z) * right_z


def _camera_line_of_sight_blocked(
    *,
    physics_world: Any | None,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> bool:
    if physics_world is None or not hasattr(physics_world, "rayTestAll"):
        return False
    core = cast(Any, import_module("panda3d.core"))
    start_x, start_y, start_z = start
    end_x, end_y, end_z = end
    result = physics_world.rayTestAll(
        core.Vec3(start_x, start_y + LIDAR_SENSOR_HEIGHT_M, start_z),
        core.Vec3(end_x, end_y + LIDAR_SENSOR_HEIGHT_M, end_z),
    )
    if not hasattr(result, "getHits"):
        return False
    for hit in result.getHits():
        node_name = _node_name(hit.getNode()) if hasattr(hit, "getNode") else ""
        hit_fraction = float(hit.getHitFraction()) if hasattr(hit, "getHitFraction") else 1.0
        if node_name.startswith("track-barrier") and hit_fraction < 0.99:
            return True
    return False


def _rate_per_second(delta: float, dt_s: float, *, enabled: bool) -> float:
    if not enabled or dt_s <= 0.0:
        return 0.0
    return delta / dt_s


def _continuous_contact_seconds(*, previous_seconds: float, active: bool, dt_s: float) -> float:
    if not active:
        return 0.0
    return max(0.0, previous_seconds) + max(0.0, dt_s)


def _tuple_value(values: tuple[float, ...], index: int, *, default: float) -> float:
    if index < len(values):
        return values[index]
    return default


def _robot_speed_mps(robot: RobotVehicle) -> float:
    return float(robot.vehicle.getCurrentSpeedKmHour()) / 3.6


def _node_heading_degrees(node_path: Any) -> float:
    return float(node_path.getH()) if hasattr(node_path, "getH") else 0.0


def _node_pitch_degrees(node_path: Any) -> float:
    return float(node_path.getP()) if hasattr(node_path, "getP") else 0.0


def _node_roll_degrees(node_path: Any) -> float:
    return float(node_path.getR()) if hasattr(node_path, "getR") else 0.0


def _node_name(node: Any) -> str:
    return str(node.getName()) if hasattr(node, "getName") else ""
