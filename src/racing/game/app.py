"""Build the graphical simulator scenes and connect them to physics and input."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from racing.game.config import (
    CameraView,
    CarShowcaseConfig,
    GameConfig,
    HeadToHeadViewerConfig,
    RunnableApp,
    configure_window,
    fps_text_for_delta,
)
from racing.sound.audio import AudioKeyToggleState, RacingAudioRuntimeLike, create_racing_audio_runtime, update_audio_mute_key
from racing.graphics.camera import (
    FORMULA_DRONE_CAMERA_SETTINGS,
    FORMULA_FOLLOW_CAMERA_SETTINGS,
    CameraRig,
    FollowCameraSettings,
    apply_camera_view,
    update_camera_cycle,
)
from racing.student.api import RobotCommand
from racing.controls.gamepad import sync_gamepad_axes
from racing.race.head_to_head import (
    HeadToHeadRaceEntry,
    HeadToHeadRaceResult,
    HeadToHeadResult,
    classify_head_to_head_winner,
    format_head_to_head_result,
    head_to_head_race_entries,
    head_to_head_race_margin,
    head_to_head_team_stats_from_runtimes,
)
from racing.controls.keyboard import manual_drive_command
from racing.graphics.lighting import add_lighting, add_showcase_lighting
from racing.graphics.panda_config import (
    configure_panda_antialiasing,
    configure_panda_y_up,
    enable_render_antialiasing,
    patch_ursina_window_coordinate_system,
    quiet_panda_image_logs,
)
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    RobotVehicle,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
    vehicle_spawn_height,
)
from racing.race.progress import (
    TrackProjection,
    default_track_progress_model,
    project_track_position,
)
from racing.race.rules import HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M, HeadToHeadRaceRules
from racing.race.runtime import (
    RaceCarRuntime,
    RaceContactState,
    RaceRecoveryConfig,
    RaceSpawnPose,
    lap_progress_tracker_for_spawn_pose,
    maybe_marshal_race_runtimes,
    quit_ursina_app,
    race_contact_states,
    race_spawn_poses,
    reset_robot_vehicle,
    robot_score_damage,
    robot_track_point,
    seeded_race_start_finish_pose,
    start_finish_pose_for_progress,
    update_race_runtime_after_step,
)
from racing.graphics.render_assets import create_scene_assets
from racing.race.sensors import RobotSensorBuilderState, build_robot_sensors
from racing.graphics.track_rendering import (
    NIGHT_SKY_COLOR,
    START_HEADING_DEGREES,
    TRACK_SURFACE_Y,
    add_mugello_short_track,
    add_racing_scene_collisions,
    add_trackside_scenery,
    add_world_floor,
    set_start_finish_gantry_pose,
    set_start_finish_pose,
    start_finish_render_pose,
)
from racing.graphics.vehicle_visuals import (
    add_robot_visuals,
    add_showcase_floor,
    apply_robot_team_color,
    apply_showcase_camera,
    create_showcase_robot,
    pose_showcase_car,
)
from racing.track.world import START_POSITION, TrackPoint

PLAYABLE_MAX_FRAME_DELTA_SECONDS = 0.25
PLAYABLE_MAX_FIXED_STEPS_PER_FRAME = 8
DAMAGE_HUD_MAX_COLUMNS = 4
DAMAGE_HUD_USABLE_WIDTH = 3.30
DAMAGE_HUD_COLUMN_GAP = 0.055
DAMAGE_HUD_MAX_WIDTH = 0.78
DAMAGE_HUD_MIN_WIDTH = 0.42
DAMAGE_HUD_HEIGHT = 0.055
DAMAGE_HUD_BOTTOM_Y = -0.915
DAMAGE_HUD_ROW_SPACING = 0.088
DAMAGE_HUD_EMPTY_WIDTH = 0.004
DAMAGE_HUD_SHADOW_COLOR = (0.0, 0.0, 0.0, 0.62)
DAMAGE_HUD_TRACK_COLOR = (0.020, 0.023, 0.030, 0.92)
DAMAGE_HUD_FRAME_COLOR = (0.92, 0.94, 0.98, 0.80)
DAMAGE_HUD_INNER_FRAME_COLOR = (0.18, 0.20, 0.24, 0.90)
DAMAGE_HUD_ZERO_FILL_COLOR = (0.28, 0.92, 0.40, 0.92)
DAMAGE_HUD_MID_FILL_COLOR = (1.00, 0.74, 0.16, 0.96)
DAMAGE_HUD_FULL_FILL_COLOR = (1.00, 0.12, 0.08, 0.98)
DAMAGE_HUD_ELIMINATED_FILL_COLOR = (0.74, 0.0, 0.0, 1.0)

ColorRGBA = tuple[float, float, float, float]
DEFAULT_WINDOW_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "textures" / "ursina.ico"

quiet_panda_image_logs()


def _create_configured_ursina_app(*, app_kwargs: dict[str, Any], preserve_project_y_up: bool = True) -> tuple[Any, Any]:
    """Create an Ursina app after applying project-wide Panda3D window config."""
    configured_app_kwargs = {"icon": str(DEFAULT_WINDOW_ICON_PATH), **app_kwargs}
    if preserve_project_y_up:
        configure_panda_y_up()
    configure_panda_antialiasing()
    ursina = cast(Any, import_module("ursina"))

    if preserve_project_y_up:
        restore_ursina_coordinate_system = patch_ursina_window_coordinate_system()
        try:
            app = ursina.Ursina(**configured_app_kwargs)
        finally:
            restore_ursina_coordinate_system()
    else:
        app = ursina.Ursina(**configured_app_kwargs)
    enable_render_antialiasing(ursina.scene)
    return ursina, app


@dataclass(frozen=True, slots=True)
class DamageHudSlot:
    """Screen-space placement for one car damage bar."""

    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(slots=True)
class DamageHudBar:
    """Small group of 2D objects that show one car's damage."""

    slot: DamageHudSlot
    shadow: Any
    frame: Any
    track: Any
    fill: Any
    cap: Any
    accent: Any


@dataclass(slots=True)
class AudioHudControl:
    """Small on-screen audio toggle and its key state."""

    button: Any
    label: Any
    key_state: AudioKeyToggleState


def _follow_camera_settings_for_view(view: CameraView) -> FollowCameraSettings:
    if view in (CameraView.DRONE, CameraView.FOLLOW_CAR):
        return FORMULA_DRONE_CAMERA_SETTINGS
    if view is CameraView.FOLLOW:
        return FORMULA_FOLLOW_CAMERA_SETTINGS
    return FORMULA_DRONE_CAMERA_SETTINGS


def build_scene(config: GameConfig) -> RunnableApp:
    """Create the single-car scene used for manual driving or one student controller."""
    if config.fixed_delta_seconds <= 0.0:
        raise ValueError("fixed_delta_seconds must be positive")

    app_kwargs: dict[str, Any] = {
        "title": config.title,
        "borderless": config.borderless,
        "fullscreen": config.fullscreen,
        "vsync": config.vsync,
        "development_mode": config.development_mode,
        "size": config.size,
    }
    if config.window_type is not None:
        app_kwargs["window_type"] = config.window_type
    ursina, app = _create_configured_ursina_app(app_kwargs=app_kwargs)
    if config.window_type is None:
        configure_window(ursina.window, config)
    app.setBackgroundColor(*NIGHT_SKY_COLOR)
    assets = create_scene_assets()

    physics_world = create_physics_world()
    physics_scene = PhysicsScene(world=physics_world, vehicles=[])

    track_model = default_track_progress_model()
    spawn_position = (
        config.spawn_position
        if config.spawn_position is not None
        else (
            START_POSITION.x,
            vehicle_spawn_height(FORMULA_VEHICLE_PHYSICS_CONFIG, surface_y=TRACK_SURFACE_Y),
            START_POSITION.z,
        )
    )
    spawn_heading_degrees = (
        START_HEADING_DEGREES if config.spawn_heading_degrees is None else config.spawn_heading_degrees
    )
    spawn_progress_distance_m = (
        project_track_position(track_model, TrackPoint(spawn_position[0], spawn_position[2])).progress_distance_m
        if config.spawn_progress_distance_m is None
        else config.spawn_progress_distance_m
    )
    start_finish_progress_pose = start_finish_pose_for_progress(
        model=track_model,
        start_progress_distance_m=spawn_progress_distance_m,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )
    start_finish_pose = start_finish_render_pose(
        position=start_finish_progress_pose.position,
    )

    add_world_floor(ursina=ursina, physics_world=physics_world, assets=assets, include_collision=False)
    add_mugello_short_track(
        ursina=ursina,
        physics_world=physics_world,
        assets=assets,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
        include_collision=False,
    )
    add_racing_scene_collisions(physics_world=physics_world, render=ursina.scene)
    add_trackside_scenery(
        ursina=ursina,
        assets=assets,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
    )

    robot = create_robot_vehicle(
        world=physics_world,
        render=ursina.scene,
        name="student-robot-0",
        position=spawn_position,
        heading_degrees=spawn_heading_degrees,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )
    physics_scene.vehicles.append(robot)
    app.racing_robot = robot
    add_robot_visuals(ursina=ursina, robot=robot, assets=assets, team_color=config.team_color)

    student_runtime = (
        student_marshal_runtime(
            robot=robot,
            start_position=TrackPoint(spawn_position[0], spawn_position[2]),
            starting_progress_distance_m=spawn_progress_distance_m,
        )
        if config.student_controller is not None
        else None
    )
    student_recovery_config = (
        _head_to_head_viewer_recovery_config(HeadToHeadRaceRules()) if student_runtime is not None else None
    )

    add_lighting(ursina)
    camera_rig = CameraRig(view=config.camera_view)
    apply_camera_view(
        ursina=ursina,
        view=camera_rig.view,
        target=robot.chassis_np,
        rig=camera_rig,
        follow_settings=_follow_camera_settings_for_view(camera_rig.view),
        track_model=track_model,
    )

    fps_display = ursina.Text(text="FPS: --", position=(-0.87, 0.46), scale=0.85, background=True)
    speed_display = ursina.Text(text="0.0 km/h", position=(-0.87, 0.40), scale=0.75, background=True)
    damage_bars = _add_damage_hud_bars(ursina=ursina, colors=(config.team_color,))
    audio_runtime = create_racing_audio_runtime(ursina=ursina, config=config.audio)
    _register_audio_vehicles(audio_runtime=audio_runtime, robots=(robot,))
    audio_control = _add_audio_hud_control(ursina=ursina, audio_runtime=audio_runtime)
    sensor_state = RobotSensorBuilderState()
    simulation_time_s = 0.0
    simulation_accumulator_seconds = 0.0

    def update() -> None:
        """Advance the playable scene by one rendered frame."""
        nonlocal sensor_state, simulation_accumulator_seconds, simulation_time_s
        frame_delta_seconds = min(float(ursina.time.dt), PLAYABLE_MAX_FRAME_DELTA_SECONDS)
        update_camera_cycle(camera_rig, cycle_key_down=bool(ursina.held_keys["v"]))
        _update_audio_key_control(
            audio_control=audio_control, audio_runtime=audio_runtime, mute_key_down=bool(ursina.held_keys["m"])
        )

        simulation_accumulator_seconds += frame_delta_seconds
        fixed_steps = 0
        while (
            simulation_accumulator_seconds >= config.fixed_delta_seconds
            and fixed_steps < PLAYABLE_MAX_FIXED_STEPS_PER_FRAME
        ):
            simulation_time_s += config.fixed_delta_seconds
            if not robot.eliminated:
                if config.student_controller is None:
                    sync_gamepad_axes(ursina.held_keys)
                    command = manual_drive_command(ursina.held_keys)
                else:
                    sensors, sensor_state = build_robot_sensors(
                        physics_world=physics_world,
                        robot=robot,
                        track_model=track_model,
                        time_s=simulation_time_s,
                        dt_s=config.fixed_delta_seconds,
                        previous_state=sensor_state,
                    )
                    command = config.student_controller(sensors)
                audio_runtime.record_command(robot, command)
                apply_robot_vehicle_command(robot=robot, command=command)
            physics_scene.step(config.fixed_delta_seconds)
            simulation_accumulator_seconds -= config.fixed_delta_seconds
            fixed_steps += 1

            student_contact_state: RaceContactState | None = None
            student_projection: TrackProjection | None = None
            if student_runtime is not None and student_recovery_config is not None:
                student_contact_state = race_contact_states(physics_world=physics_world, runtimes=(student_runtime,))[0]
                student_projection = project_track_position(track_model, robot_track_point(robot))
            apply_wall_impact_damage(
                physics_world=physics_world,
                robots=(robot,),
                fixed_time_step=physics_scene.fixed_time_step,
            )
            if (
                student_runtime is not None
                and student_recovery_config is not None
                and student_contact_state is not None
                and student_projection is not None
            ):
                update_race_runtime_after_step(
                    runtime=student_runtime,
                    projection=student_projection,
                    contact_state=student_contact_state,
                    elapsed_seconds=simulation_time_s,
                    delta_seconds=config.fixed_delta_seconds,
                )
                if (
                    not robot.eliminated
                    and maybe_marshal_race_runtimes(
                        runtimes=(student_runtime,),
                        projections=(student_projection,),
                        recovery_config=student_recovery_config,
                        delta_seconds=config.fixed_delta_seconds,
                    )
                    > 0
                ):
                    sensor_state = student_sensor_state_after_marshal(
                        previous_state=sensor_state,
                        projection=student_projection,
                        time_s=simulation_time_s,
                    )
                    camera_rig.reset_follow_history()

        if fixed_steps == PLAYABLE_MAX_FIXED_STEPS_PER_FRAME:
            simulation_accumulator_seconds = min(simulation_accumulator_seconds, config.fixed_delta_seconds)

        apply_camera_view(
            ursina=ursina,
            view=camera_rig.view,
            target=robot.chassis_np,
            rig=camera_rig,
            delta_seconds=frame_delta_seconds,
            follow_settings=_follow_camera_settings_for_view(camera_rig.view),
            track_model=track_model,
        )
        audio_runtime.update(frame_delta_seconds)
        _sync_audio_hud_control(audio_control=audio_control, audio_runtime=audio_runtime)
        fps_display.text = fps_text_for_delta(frame_delta_seconds)
        speed_display.text = (
            "OUT" if robot.eliminated else f"{abs(float(robot.vehicle.getCurrentSpeedKmHour())):>4.1f} km/h"
        )
        _update_damage_hud_bars(bars=damage_bars, robots=(robot,))

    ursina.Entity(name="simulation_loop", update=update, ignore_paused=True)
    return cast(RunnableApp, app)


def student_marshal_runtime(
    *,
    robot: RobotVehicle,
    start_position: TrackPoint,
    starting_progress_distance_m: float | None = None,
) -> RaceCarRuntime:
    """Create the race bookkeeping needed to reset a stuck student car."""
    model = default_track_progress_model()
    start_projection = project_track_position(model, start_position)
    tracker = lap_progress_tracker_for_spawn_pose(
        model=model,
        spawn_pose=RaceSpawnPose(
            position=(start_position.x, 0.0, start_position.z),
            heading_degrees=start_projection.heading_degrees,
            progress_distance_m=(
                start_projection.progress_distance_m
                if starting_progress_distance_m is None
                else starting_progress_distance_m
            ),
        ),
    )
    return RaceCarRuntime(robot=robot, tracker=tracker)


def student_sensor_state_after_marshal(
    *,
    previous_state: RobotSensorBuilderState,
    projection: TrackProjection,
    time_s: float,
) -> RobotSensorBuilderState:
    """Reset sensor bookkeeping after the marshal moves a student car."""
    return RobotSensorBuilderState(
        time_s=time_s,
        position=projection.nearest_center,
        heading_degrees=projection.heading_degrees,
        speed_mps=0.0,
        distance_m=previous_state.distance_m,
        tick=previous_state.tick,
    )


def build_head_to_head_viewer_scene(config: HeadToHeadViewerConfig) -> RunnableApp:
    """Create the visual race viewer for two controller teams."""
    _validate_head_to_head_viewer_config(config)
    race_rules = _head_to_head_viewer_rules(config)
    recovery_config = _head_to_head_viewer_recovery_config(race_rules)

    app_kwargs: dict[str, Any] = {
        "title": config.title,
        "borderless": config.borderless,
        "fullscreen": config.fullscreen,
        "vsync": config.vsync,
        "development_mode": config.development_mode,
        "size": config.size,
    }
    if config.window_type is not None:
        app_kwargs["window_type"] = config.window_type
    ursina, app = _create_configured_ursina_app(app_kwargs=app_kwargs)
    if config.window_type is None:
        configure_window(ursina.window, config)
    app.setBackgroundColor(*NIGHT_SKY_COLOR)
    assets = create_scene_assets()

    physics_world = create_physics_world()
    physics_scene = PhysicsScene(world=physics_world, vehicles=[])

    add_world_floor(ursina=ursina, physics_world=physics_world, assets=assets, include_collision=False)
    model = default_track_progress_model()
    start_finish_progress_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=config.random_seed,
        race_index=1,
    )
    start_finish_pose = start_finish_render_pose(
        position=start_finish_progress_pose.position,
    )
    start_finish_track_line = add_mugello_short_track(
        ursina=ursina,
        physics_world=physics_world,
        assets=assets,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
        include_collision=False,
    )
    add_racing_scene_collisions(physics_world=physics_world, render=ursina.scene)
    start_finish_gantry = add_trackside_scenery(
        ursina=ursina,
        assets=assets,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
    )

    entries = head_to_head_race_entries(
        challenger_copies=config.challenger_copies,
        incumbent_copies=config.incumbent_copies,
        race_index=1,
        random_seed=config.random_seed,
    )
    spawn_poses = race_spawn_poses(
        len(entries),
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=config.random_seed,
        race_index=1,
    )
    runtimes: list[RaceCarRuntime] = []
    team_markers: list[Any | None] = []
    for index, (entry, spawn_pose) in enumerate(zip(entries, spawn_poses, strict=True)):
        robot = create_robot_vehicle(
            world=physics_world,
            render=ursina.scene,
            name=f"h2h-robot-{entry.role}-{entry.copy_index}-{index}",
            position=spawn_pose.position,
            heading_degrees=spawn_pose.heading_degrees,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        )
        physics_scene.vehicles.append(robot)
        add_robot_visuals(
            ursina=ursina,
            robot=robot,
            assets=assets,
            team_color=_head_to_head_car_paint_color(config=config, entry=entry),
        )
        team_marker, label = _add_head_to_head_car_label(ursina=ursina, robot=robot, config=config, entry=entry)
        team_markers.append(team_marker)
        _style_head_to_head_label(label=label, config=config, entry=entry)
        runtimes.append(
            RaceCarRuntime(
                robot=robot,
                tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose),
                label=label,
            )
        )

    add_lighting(ursina)
    camera_rig = CameraRig(view=config.camera_view)
    initial_camera_target_runtime = _head_to_head_camera_target_runtime(
        config=config, entries=entries, runtimes=tuple(runtimes)
    )
    apply_camera_view(
        ursina=ursina,
        view=camera_rig.view,
        target=initial_camera_target_runtime.robot.chassis_np,
        rig=camera_rig,
        follow_settings=_follow_camera_settings_for_view(camera_rig.view),
        track_model=model,
    )

    status_display = ursina.Text(text="", position=(-0.88, 0.46), scale=0.60, color=(1, 1, 1, 1), background=True)
    race_display = ursina.Text(text="", position=(-0.88, 0.38), scale=0.50, color=(1, 1, 1, 1), background=True)
    damage_bars = _add_damage_hud_bars(
        ursina=ursina, colors=tuple(_head_to_head_team_color(config=config, role=entry.role) for entry in entries)
    )
    audio_runtime = create_racing_audio_runtime(ursina=ursina, config=config.audio)
    _register_audio_vehicles(audio_runtime=audio_runtime, robots=tuple(runtime.robot for runtime in runtimes))
    audio_control = _add_audio_hud_control(ursina=ursina, audio_runtime=audio_runtime)

    race_index = 1
    race_elapsed_seconds = 0.0
    simulation_accumulator_seconds = 0.0
    completed_race_results: list[HeadToHeadRaceResult] = []

    def start_race(next_race_index: int) -> tuple[HeadToHeadRaceEntry, ...]:
        """Reset cars, labels, and start/finish art for the next race."""
        nonlocal race_elapsed_seconds
        race_elapsed_seconds = 0.0
        next_entries = head_to_head_race_entries(
            challenger_copies=config.challenger_copies,
            incumbent_copies=config.incumbent_copies,
            race_index=next_race_index,
            random_seed=config.random_seed,
        )
        next_spawn_poses = race_spawn_poses(
            len(runtimes),
            model=model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=config.random_seed,
            race_index=next_race_index,
        )
        next_start_finish_progress_pose = seeded_race_start_finish_pose(
            model=model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=config.random_seed,
            race_index=next_race_index,
        )
        next_start_finish_pose = start_finish_render_pose(
            position=next_start_finish_progress_pose.position,
        )
        set_start_finish_pose(
            start_finish_track_line,
            position=next_start_finish_pose.position,
            heading_degrees=next_start_finish_pose.heading_degrees,
        )
        set_start_finish_gantry_pose(
            start_finish_gantry,
            position=next_start_finish_pose.position,
            heading_degrees=next_start_finish_pose.heading_degrees,
        )
        for index, (runtime, team_marker, entry, spawn_pose) in enumerate(
            zip(runtimes, team_markers, next_entries, next_spawn_poses, strict=True)
        ):
            runtime.tracker = lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose)
            runtime.stuck_seconds = 0.0
            runtime.low_progress_seconds = 0.0
            runtime.off_track_seconds = 0.0
            runtime.recent_progress_mps = 0.0
            runtime.max_speed_mps = 0.0
            runtime.contact_state = RaceContactState()
            runtime.sensor_state = RobotSensorBuilderState()
            runtime.marshal_count = 0
            runtime.marshal_penalty_m = 0.0
            runtime.marshal_cooldown_seconds = 0.0
            reset_robot_vehicle(
                runtime.robot,
                position=spawn_pose.position,
                heading_degrees=spawn_pose.heading_degrees,
                reset_damage=True,
            )
            team_color = _head_to_head_team_color(config=config, role=entry.role)
            apply_robot_team_color(
                robot=runtime.robot, assets=assets, team_color=_head_to_head_car_paint_color(config=config, entry=entry)
            )
            if team_marker is not None:
                team_marker.color = team_color
            _style_damage_hud_bar(bar=damage_bars[index], color=team_color)
            _style_head_to_head_label(label=runtime.label, config=config, entry=entry)
            runtime.robot.chassis_np.setName(f"h2h-robot-{entry.role}-{entry.copy_index}-{index}")
        return next_entries

    def update() -> None:
        """Advance the head-to-head viewer by one rendered frame."""
        nonlocal entries, race_elapsed_seconds, race_index, simulation_accumulator_seconds
        frame_delta_seconds = min(float(ursina.time.dt), 0.25)
        update_camera_cycle(camera_rig, cycle_key_down=bool(ursina.held_keys["v"]))
        _update_audio_key_control(
            audio_control=audio_control, audio_runtime=audio_runtime, mute_key_down=bool(ursina.held_keys["m"])
        )

        simulation_accumulator_seconds += frame_delta_seconds
        while simulation_accumulator_seconds >= config.fixed_delta_seconds:
            for entry, runtime in zip(entries, runtimes, strict=True):
                command = _head_to_head_viewer_command(
                    config=config,
                    entry=entry,
                    model=model,
                    runtime=runtime,
                    physics_world=physics_world,
                    runtimes=tuple(runtimes),
                    time_s=race_elapsed_seconds,
                    dt_s=config.fixed_delta_seconds,
                    held_keys=ursina.held_keys,
                )
                audio_runtime.record_command(runtime.robot, command)
                apply_robot_vehicle_command(robot=runtime.robot, command=command)

            physics_scene.step(config.fixed_delta_seconds)
            race_elapsed_seconds += config.fixed_delta_seconds
            simulation_accumulator_seconds -= config.fixed_delta_seconds

            contact_states = race_contact_states(physics_world=physics_world, runtimes=tuple(runtimes))
            apply_wall_impact_damage(
                physics_world=physics_world,
                robots=tuple(runtime.robot for runtime in runtimes),
                fixed_time_step=physics_scene.fixed_time_step,
            )
            projections: list[TrackProjection] = []
            for runtime, contact_state in zip(runtimes, contact_states, strict=True):
                projection = project_track_position(model, robot_track_point(runtime.robot))
                projections.append(projection)
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=race_elapsed_seconds,
                    delta_seconds=config.fixed_delta_seconds,
                )
            if recovery_config is not None:
                maybe_marshal_race_runtimes(
                    runtimes=tuple(runtimes),
                    projections=tuple(projections),
                    recovery_config=recovery_config,
                    delta_seconds=config.fixed_delta_seconds,
                )

            if race_elapsed_seconds >= config.round_seconds:
                break

        camera_target_runtime = _head_to_head_camera_target_runtime(
            config=config, entries=entries, runtimes=tuple(runtimes)
        )
        apply_camera_view(
            ursina=ursina,
            view=camera_rig.view,
            target=camera_target_runtime.robot.chassis_np,
            rig=camera_rig,
            delta_seconds=frame_delta_seconds,
            follow_settings=_follow_camera_settings_for_view(camera_rig.view),
            track_model=model,
        )
        audio_runtime.update(frame_delta_seconds)
        _sync_audio_hud_control(audio_control=audio_control, audio_runtime=audio_runtime)
        _update_head_to_head_hud(
            status_display=status_display,
            race_display=race_display,
            config=config,
            race_rules=race_rules,
            race_index=race_index,
            entries=entries,
            runtimes=tuple(runtimes),
            race_elapsed_seconds=race_elapsed_seconds,
        )
        _update_damage_hud_bars(bars=damage_bars, robots=tuple(runtime.robot for runtime in runtimes))

        if race_elapsed_seconds < config.round_seconds:
            return

        completed_race_results.append(
            _head_to_head_race_result_from_runtimes(
                config=config,
                race_rules=race_rules,
                race_index=race_index,
                entries=entries,
                runtimes=tuple(runtimes),
            )
        )
        if race_index >= config.race_count:
            print(
                format_head_to_head_result(
                    HeadToHeadResult(
                        challenger_name=config.challenger_name,
                        incumbent_name=config.incumbent_name,
                        round_seconds=config.round_seconds,
                        win_margin_m=race_rules.win_margin_m,
                        races=tuple(completed_race_results),
                        random_seed=config.random_seed,
                        rules=race_rules,
                    )
                )
            )
            quit_ursina_app(ursina=ursina, app=app)
            return

        race_index += 1
        entries = start_race(race_index)
        simulation_accumulator_seconds = 0.0

    ursina.Entity(name="head_to_head_viewer_loop", update=update, ignore_paused=True)
    return cast(RunnableApp, app)


def _validate_head_to_head_viewer_config(config: HeadToHeadViewerConfig) -> None:
    if config.race_count < 1:
        raise ValueError("race_count must be at least one")
    if config.round_seconds <= 0.0:
        raise ValueError("round_seconds must be positive")
    if config.fixed_delta_seconds <= 0.0:
        raise ValueError("fixed_delta_seconds must be positive")
    if config.challenger_copies < 1:
        raise ValueError("challenger_copies must be at least one")
    if config.incumbent_copies < 1:
        raise ValueError("incumbent_copies must be at least one")
    if config.challenger_keyboard and config.incumbent_keyboard:
        raise ValueError("keyboard control can only be assigned to one head-to-head side")
    if config.challenger_keyboard and config.challenger_copies != 1:
        raise ValueError("keyboard-controlled challenger must use exactly one copy")
    if config.incumbent_keyboard and config.incumbent_copies != 1:
        raise ValueError("keyboard-controlled incumbent must use exactly one copy")
    if config.challenger_keyboard and config.challenger_controller is not None:
        raise ValueError("challenger keyboard control cannot be combined with a challenger controller")
    if config.incumbent_keyboard and config.incumbent_controller is not None:
        raise ValueError("incumbent keyboard control cannot be combined with an incumbent controller")
    if not (config.challenger_keyboard or config.challenger_controller is not None):
        raise ValueError("challenger needs keyboard control or a student controller")
    if not (config.incumbent_keyboard or config.incumbent_controller is not None):
        raise ValueError("incumbent needs keyboard control or a student controller")
    _head_to_head_viewer_rules(config)


def _head_to_head_viewer_rules(config: HeadToHeadViewerConfig) -> HeadToHeadRaceRules:
    if config.win_margin_m == HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M:
        return config.rules
    return replace(config.rules, win_margin_m=config.win_margin_m)


def _head_to_head_viewer_recovery_config(rules: HeadToHeadRaceRules) -> RaceRecoveryConfig | None:
    if not rules.marshal_enabled:
        return None
    return RaceRecoveryConfig(
        stuck_seconds=rules.marshal_stuck_seconds,
        distance_penalty_m=rules.marshal_penalty_m,
        cooldown_seconds=rules.marshal_cooldown_seconds,
    )


def _head_to_head_viewer_command(
    *,
    config: HeadToHeadViewerConfig,
    entry: HeadToHeadRaceEntry,
    model: Any,
    runtime: RaceCarRuntime,
    physics_world: Any,
    runtimes: tuple[RaceCarRuntime, ...],
    time_s: float,
    dt_s: float,
    held_keys: Any,
) -> RobotCommand:
    if _head_to_head_viewer_keyboard_controlled(config=config, entry=entry):
        sync_gamepad_axes(held_keys)
        return manual_drive_command(held_keys)
    controller = _head_to_head_viewer_controller(config=config, entry=entry)
    if controller is None:
        raise ValueError("head-to-head entry has no controller")
    sensors, runtime.sensor_state = build_robot_sensors(
        physics_world=physics_world,
        robot=runtime.robot,
        track_model=model,
        time_s=time_s,
        dt_s=dt_s,
        previous_state=runtime.sensor_state,
        other_robot_node_names=_head_to_head_other_runtime_node_names(runtime=runtime, runtimes=runtimes),
        other_robots=_head_to_head_other_runtime_robots(runtime=runtime, runtimes=runtimes),
    )
    return controller(sensors)


def _head_to_head_viewer_controller(*, config: HeadToHeadViewerConfig, entry: HeadToHeadRaceEntry) -> Any:
    if entry.role == "challenger":
        return config.challenger_controller
    return config.incumbent_controller


def _head_to_head_viewer_keyboard_controlled(*, config: HeadToHeadViewerConfig, entry: HeadToHeadRaceEntry) -> bool:
    if entry.role == "challenger":
        return config.challenger_keyboard
    return config.incumbent_keyboard


def _head_to_head_camera_target_runtime(
    *,
    config: HeadToHeadViewerConfig,
    entries: tuple[HeadToHeadRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
) -> RaceCarRuntime:
    for entry, runtime in zip(entries, runtimes, strict=True):
        if _head_to_head_viewer_keyboard_controlled(config=config, entry=entry):
            return runtime
    return _leader_runtime(runtimes)


def _head_to_head_other_runtime_node_names(
    *, runtime: RaceCarRuntime, runtimes: tuple[RaceCarRuntime, ...]
) -> frozenset[str]:
    return frozenset(
        _head_to_head_runtime_node_name(other_runtime.robot.chassis_np.node())
        for other_runtime in runtimes
        if other_runtime is not runtime and not bool(getattr(other_runtime.robot, "eliminated", False))
    )


def _head_to_head_other_runtime_robots(
    *, runtime: RaceCarRuntime, runtimes: tuple[RaceCarRuntime, ...]
) -> tuple[RobotVehicle, ...]:
    return tuple(
        other_runtime.robot
        for other_runtime in runtimes
        if other_runtime is not runtime and not bool(getattr(other_runtime.robot, "eliminated", False))
    )


def _head_to_head_runtime_node_name(node: Any) -> str:
    return str(node.getName()) if hasattr(node, "getName") else ""


def _style_head_to_head_label(*, label: Any | None, config: HeadToHeadViewerConfig, entry: HeadToHeadRaceEntry) -> None:
    if label is None:
        return
    label.text = _head_to_head_car_label(config=config, entry=entry)
    label.color = _head_to_head_team_color(config=config, role=entry.role)


def _add_head_to_head_car_label(
    *, ursina: Any, robot: Any, config: HeadToHeadViewerConfig, entry: HeadToHeadRaceEntry
) -> tuple[Any | None, Any]:
    color = _head_to_head_team_color(config=config, role=entry.role)
    label = ursina.Text(
        text=_head_to_head_car_label(config=config, entry=entry),
        parent=robot.chassis_np,
        position=(0.0, 1.05, 0.0),
        scale=1.2,
        origin=(0.0, 0.0),
        color=color,
        billboard=True,
        background=False,
    )
    return None, label


def _head_to_head_car_label(*, config: HeadToHeadViewerConfig, entry: HeadToHeadRaceEntry) -> str:
    team_name = config.challenger_name if entry.role == "challenger" else config.incumbent_name
    return f"{_short_head_to_head_name(team_name, max_length=18)} {entry.copy_index + 1}"


def _head_to_head_team_color(*, config: HeadToHeadViewerConfig, role: str) -> ColorRGBA:
    if role == "challenger":
        return config.challenger_team_color
    return config.incumbent_team_color


def _head_to_head_car_paint_color(*, config: HeadToHeadViewerConfig, entry: HeadToHeadRaceEntry) -> ColorRGBA:
    return _head_to_head_team_color(config=config, role=entry.role)


def _short_head_to_head_name(name: str, *, max_length: int) -> str:
    if len(name) <= max_length:
        return name
    if max_length <= 3:
        return name[:max_length]
    return f"{name[: max_length - 3]}..."


def _leader_runtime(runtimes: tuple[RaceCarRuntime, ...]) -> RaceCarRuntime:
    active_runtimes = tuple(runtime for runtime in runtimes if not runtime.robot.eliminated)
    if not active_runtimes:
        return runtimes[0]
    return max(active_runtimes, key=lambda runtime: runtime.tracker.best_distance_m)


def _register_audio_vehicles(*, audio_runtime: RacingAudioRuntimeLike, robots: tuple[RobotVehicle, ...]) -> None:
    for robot in robots:
        audio_runtime.register_vehicle(robot)


def _add_audio_hud_control(*, ursina: Any, audio_runtime: RacingAudioRuntimeLike) -> AudioHudControl | None:
    if not audio_runtime.enabled:
        return None
    parent = ursina.application.base.aspect2d
    _panda2d_hud_card(
        parent=parent,
        name="audio-hud-background",
        position=(1.43, 0.89),
        scale=(0.46, 0.090),
        color=(0.020, 0.023, 0.030, 0.88),
        bin_order=90,
    )
    label = _panda2d_hud_text(
        parent=parent,
        name="audio-hud-label",
        text=audio_runtime.button_text(),
        position=(1.43, 0.872),
        scale=0.043,
        color=(0.96, 0.98, 1.0, 1.0),
        bin_order=91,
    )
    button_color = ursina.color.rgba(0.020, 0.023, 0.030, 0.88)
    button = ursina.Button(
        text="",
        position=(0.76, 0.44),
        scale=(0.22, 0.060),
        color=ursina.color.rgba(0.0, 0.0, 0.0, 0.0),
        highlight_color=button_color.tint(0.18),
        pressed_color=button_color.tint(0.32),
    )
    control = AudioHudControl(button=button, label=label, key_state=AudioKeyToggleState())

    def toggle_audio() -> None:
        """Toggle mute from the on-screen audio button."""
        audio_runtime.toggle_muted()
        _sync_audio_hud_control(audio_control=control, audio_runtime=audio_runtime)

    button.on_click = toggle_audio
    return control


def _update_audio_key_control(
    *, audio_control: AudioHudControl | None, audio_runtime: RacingAudioRuntimeLike, mute_key_down: bool
) -> None:
    if audio_control is None:
        return
    if update_audio_mute_key(audio_control.key_state, mute_key_down=mute_key_down, audio_runtime=audio_runtime):
        _sync_audio_hud_control(audio_control=audio_control, audio_runtime=audio_runtime)


def _sync_audio_hud_control(*, audio_control: AudioHudControl | None, audio_runtime: RacingAudioRuntimeLike) -> None:
    if audio_control is not None:
        _set_panda2d_hud_text(audio_control.label, audio_runtime.button_text())


def damage_hud_layout(count: int) -> tuple[DamageHudSlot, ...]:
    """Place one compact damage bar for each visible car."""
    if count < 0:
        raise ValueError("count cannot be negative")
    if count == 0:
        return ()
    column_count = min(count, DAMAGE_HUD_MAX_COLUMNS)
    slot_width = min(
        DAMAGE_HUD_MAX_WIDTH,
        max(
            DAMAGE_HUD_MIN_WIDTH, (DAMAGE_HUD_USABLE_WIDTH - DAMAGE_HUD_COLUMN_GAP * (column_count - 1)) / column_count
        ),
    )
    slots: list[DamageHudSlot] = []
    for row_index, row_start in enumerate(range(0, count, DAMAGE_HUD_MAX_COLUMNS)):
        row_count = min(DAMAGE_HUD_MAX_COLUMNS, count - row_start)
        row_width = row_count * slot_width + (row_count - 1) * DAMAGE_HUD_COLUMN_GAP
        first_center_x = -row_width / 2.0 + slot_width / 2.0
        center_y = DAMAGE_HUD_BOTTOM_Y + row_index * DAMAGE_HUD_ROW_SPACING
        for column_index in range(row_count):
            slots.append(
                DamageHudSlot(
                    center_x=first_center_x + column_index * (slot_width + DAMAGE_HUD_COLUMN_GAP),
                    center_y=center_y,
                    width=slot_width,
                    height=DAMAGE_HUD_HEIGHT,
                )
            )
    return tuple(slots)


def _add_damage_hud_bars(*, ursina: Any, colors: tuple[ColorRGBA, ...]) -> tuple[DamageHudBar, ...]:
    parent = ursina.application.base.aspect2d
    bars: list[DamageHudBar] = []
    for slot, color in zip(damage_hud_layout(len(colors)), colors, strict=True):
        shadow = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-shadow",
            position=(slot.center_x, slot.center_y - 0.006),
            scale=(slot.width + 0.050, slot.height + 0.030),
            color=DAMAGE_HUD_SHADOW_COLOR,
            bin_order=80,
        )
        frame = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-frame",
            position=(slot.center_x, slot.center_y),
            scale=(slot.width + 0.026, slot.height + 0.018),
            color=DAMAGE_HUD_FRAME_COLOR,
            bin_order=81,
        )
        track = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-track",
            position=(slot.center_x, slot.center_y),
            scale=(slot.width, slot.height),
            color=DAMAGE_HUD_TRACK_COLOR,
            bin_order=82,
        )
        fill = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-fill",
            position=(slot.center_x, slot.center_y),
            scale=(DAMAGE_HUD_EMPTY_WIDTH, slot.height - 0.018),
            color=DAMAGE_HUD_ZERO_FILL_COLOR,
            bin_order=83,
        )
        fill.hide()
        cap = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-cap",
            position=(slot.center_x, slot.center_y),
            scale=(0.010, slot.height - 0.010),
            color=(1.0, 1.0, 1.0, 0.88),
            bin_order=84,
        )
        cap.hide()
        accent = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-accent",
            position=(slot.center_x - slot.width / 2.0 - 0.026, slot.center_y),
            scale=(0.018, slot.height + 0.020),
            color=color,
            bin_order=85,
        )
        bar = DamageHudBar(slot=slot, shadow=shadow, frame=frame, track=track, fill=fill, cap=cap, accent=accent)
        _style_damage_hud_bar(bar=bar, color=color)
        bars.append(bar)
    return tuple(bars)


def _style_damage_hud_bar(*, bar: DamageHudBar, color: ColorRGBA) -> None:
    bar.accent.setColor(*color)
    bar.shadow.setColor(*DAMAGE_HUD_SHADOW_COLOR)
    bar.frame.setColor(*DAMAGE_HUD_FRAME_COLOR)
    bar.track.setColor(*DAMAGE_HUD_TRACK_COLOR)


def _update_damage_hud_bars(*, bars: tuple[DamageHudBar, ...], robots: tuple[RobotVehicle, ...]) -> None:
    for bar, robot in zip(bars, robots, strict=True):
        damage = 1.0 if robot.eliminated else _clamp01(robot.damage)
        fill_width = max(DAMAGE_HUD_EMPTY_WIDTH, (bar.slot.width - 0.024) * damage)
        fill_left_x = bar.slot.center_x - (bar.slot.width - 0.024) / 2.0
        if damage > 0.0 or robot.eliminated:
            bar.fill.show()
            bar.cap.show()
        else:
            bar.fill.hide()
            bar.cap.hide()
        bar.fill.setScale(fill_width, bar.slot.height - 0.018, 1.0)
        bar.fill.setPos(fill_left_x + fill_width / 2.0, bar.slot.center_y, 0.0)
        bar.fill.setColor(*damage_hud_fill_color(damage=damage, eliminated=robot.eliminated))
        bar.cap.setPos(fill_left_x + fill_width, bar.slot.center_y, 0.0)
        bar.cap.setColor(*(1.0, 0.94, 0.80, 0.95) if not robot.eliminated else (0.18, 0.0, 0.0, 1.0))
        bar.frame.setColor(*(DAMAGE_HUD_INNER_FRAME_COLOR if robot.eliminated else DAMAGE_HUD_FRAME_COLOR))


def _panda2d_hud_card(
    *,
    parent: Any,
    name: str,
    position: tuple[float, float],
    scale: tuple[float, float],
    color: ColorRGBA,
    bin_order: int,
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    card_maker = core.CardMaker(name)
    card_maker.setFrame(-0.5, 0.5, -0.5, 0.5)
    card = parent.attachNewNode(card_maker.generate())
    card.setPos(position[0], position[1], 0.0)
    card.setScale(scale[0], scale[1], 1.0)
    card.setColor(*color)
    card.setTransparency(core.TransparencyAttrib.MAlpha)
    card.setDepthTest(False)
    card.setDepthWrite(False)
    card.setBin("fixed", bin_order)
    card.setLightOff(1)
    return card


def _panda2d_hud_text(
    *, parent: Any, name: str, text: str, position: tuple[float, float], scale: float, color: ColorRGBA, bin_order: int
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    text_node = core.TextNode(name)
    text_node.setText(text)
    text_node.setAlign(core.TextNode.ACenter)
    text_node.setTextColor(*color)
    text_path = parent.attachNewNode(text_node)
    text_path.setPos(position[0], position[1], 0.0)
    text_path.setScale(scale)
    text_path.setDepthTest(False)
    text_path.setDepthWrite(False)
    text_path.setBin("fixed", bin_order)
    text_path.setLightOff(1)
    return text_path


def _set_panda2d_hud_text(text_path: Any, text: str) -> None:
    node = text_path.node()
    if hasattr(node, "setText"):
        node.setText(text)


def damage_hud_fill_color(*, damage: float, eliminated: bool) -> ColorRGBA:
    """Choose the damage bar color for a car's current damage state."""
    if eliminated:
        return DAMAGE_HUD_ELIMINATED_FILL_COLOR
    normalized_damage = _clamp01(damage)
    if normalized_damage <= 0.5:
        return _interpolate_color(DAMAGE_HUD_ZERO_FILL_COLOR, DAMAGE_HUD_MID_FILL_COLOR, normalized_damage * 2.0)
    return _interpolate_color(DAMAGE_HUD_MID_FILL_COLOR, DAMAGE_HUD_FULL_FILL_COLOR, (normalized_damage - 0.5) * 2.0)


def _interpolate_color(start: ColorRGBA, end: ColorRGBA, amount: float) -> ColorRGBA:
    clamped_amount = _clamp01(amount)
    return (
        start[0] + (end[0] - start[0]) * clamped_amount,
        start[1] + (end[1] - start[1]) * clamped_amount,
        start[2] + (end[2] - start[2]) * clamped_amount,
        start[3] + (end[3] - start[3]) * clamped_amount,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _update_head_to_head_hud(
    *,
    status_display: Any,
    race_display: Any,
    config: HeadToHeadViewerConfig,
    race_rules: HeadToHeadRaceRules,
    race_index: int,
    entries: tuple[HeadToHeadRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    race_elapsed_seconds: float,
) -> None:
    time_left = max(0.0, config.round_seconds - race_elapsed_seconds)
    challenger_distance = _head_to_head_scored_distance(
        entries=entries, runtimes=runtimes, role="challenger", race_rules=race_rules
    )
    incumbent_distance = _head_to_head_scored_distance(
        entries=entries, runtimes=runtimes, role="incumbent", race_rules=race_rules
    )
    status_display.text = (
        f"Race {race_index}/{config.race_count}   "
        f"Time {time_left:04.1f}s   "
        f"{_short_head_to_head_name(config.challenger_name, max_length=18)} {challenger_distance:05.1f}m   "
        f"{_short_head_to_head_name(config.incumbent_name, max_length=18)} {incumbent_distance:05.1f}m"
    )
    race_display.text = "   ".join(
        f"{_head_to_head_car_label(config=config, entry=entry)}: "
        f"{runtime.tracker.lap_count}L {_head_to_head_runtime_score(runtime):05.1f}m "
        f"W{runtime.tracker.wall_contact_seconds:03.1f} C{runtime.tracker.car_contact_seconds:03.1f} "
        f"S{runtime.low_progress_seconds:03.1f} M{runtime.marshal_count}"
        for entry, runtime in zip(entries, runtimes, strict=True)
    )


def _head_to_head_scored_distance(
    *,
    entries: tuple[HeadToHeadRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    role: str,
    race_rules: HeadToHeadRaceRules,
) -> float:
    distances = tuple(
        _head_to_head_runtime_score(runtime)
        for entry, runtime in zip(entries, runtimes, strict=True)
        if entry.role == role
    )
    if race_rules.scoring == "team-sum":
        return sum(distances)
    return max(distances)


def _head_to_head_runtime_score(runtime: RaceCarRuntime) -> float:
    raw_score = max(0.0, runtime.tracker.best_distance_m - runtime.marshal_penalty_m)
    return raw_score * (1.0 - robot_score_damage(runtime.robot))


def _head_to_head_race_result_from_runtimes(
    *,
    config: HeadToHeadViewerConfig,
    race_rules: HeadToHeadRaceRules,
    race_index: int,
    entries: tuple[HeadToHeadRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
) -> HeadToHeadRaceResult:
    challenger_stats = head_to_head_team_stats_from_runtimes(entries=entries, runtimes=runtimes, role="challenger")
    incumbent_stats = head_to_head_team_stats_from_runtimes(entries=entries, runtimes=runtimes, role="incumbent")
    winner = classify_head_to_head_winner(
        margin_m=head_to_head_race_margin(
            challenger=challenger_stats, incumbent=incumbent_stats, scoring=race_rules.scoring
        ),
        win_margin_m=race_rules.win_margin_m,
    )
    return HeadToHeadRaceResult(
        race_index=race_index,
        winner=winner,
        challenger=challenger_stats,
        incumbent=incumbent_stats,
        scoring=race_rules.scoring,
    )


def create_app(config: GameConfig | None = None) -> RunnableApp:
    """Create the normal playable simulator app."""
    return build_scene(GameConfig() if config is None else config)


def create_head_to_head_viewer_app(config: HeadToHeadViewerConfig | None = None) -> RunnableApp:
    """Create the app that watches two controller teams race."""
    return build_head_to_head_viewer_scene(HeadToHeadViewerConfig() if config is None else config)


def create_car_showcase_app(config: CarShowcaseConfig | None = None) -> RunnableApp:
    """Create the small scene used to inspect the car model."""
    showcase_config = CarShowcaseConfig() if config is None else config
    ursina, app = _create_configured_ursina_app(
        app_kwargs={
            "title": showcase_config.title,
            "borderless": False,
            "fullscreen": False,
            "vsync": False,
            "development_mode": showcase_config.development_mode,
            "editor_ui_enabled": False,
            "size": showcase_config.size,
            "window_type": showcase_config.window_type,
        },
        preserve_project_y_up=False,
    )
    app.setBackgroundColor(0.96, 0.96, 0.94, 1)

    assets = create_scene_assets()
    add_showcase_floor(ursina=ursina)
    add_showcase_lighting(ursina)

    robot = create_showcase_robot(ursina, config=FORMULA_VEHICLE_PHYSICS_CONFIG)
    add_robot_visuals(ursina=ursina, robot=robot, assets=assets, team_color=showcase_config.team_color)
    pose_showcase_car(robot.chassis_np, showcase_config.view)
    apply_showcase_camera(ursina=ursina, view=showcase_config.view)
    return cast(RunnableApp, app)
