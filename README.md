---
author: Kris Jordan
---

# Getting Started

This guide prepares you for writing the control intelligence for your own simulated, self-driving race car.

The simulator is a Python 3.11 project managed with `uv`. You will run
everything from the project folder, and `uv` can install the correct Python
version for this project.

## 1. Host Machine Requirements

You need:

- `uv`, the Python project and package manager
- A terminal
- This repository on your machine

Check that `uv` is installed:

```bash
uv --version
```

You do not need to install Python 3.11 yourself. The project requires Python
`>=3.11,<3.12`, and `uv` can download and use a matching Python version for this
project.

If `uv` is missing, install it from the official uv installation instructions:

https://docs.astral.sh/uv/getting-started/installation/

## 2. Let uv Install Python And The Project

From the project folder, check the local Python version pin:

```bash
cat .python-version
```

It should say:

```text
3.11
```

That file tells `uv` which Python version this project should use. Now ask
`uv` to use a uv-managed Python and install the project dependencies:

```bash
uv sync --managed-python
```

If Python 3.11 is not already available through `uv`, this command downloads a
matching Python version. It also creates or updates the local project
environment in `.venv/` and installs the simulator dependencies there.

You do not need to activate `.venv/`; use `uv run ...` from this folder and
`uv` will use the project environment automatically.

Check the Python version inside the project environment:

```bash
uv run python --version
```

It should print Python `3.11.x`.

You can check that the command-line entry point is available with:

```bash
uv run racing --help
```

## 3. Drive The Keyboard Demo

Start the simulator with manual keyboard control:

```bash
uv run racing
```

Keyboard controls:

- Up arrow: accelerate
- Down arrow: brake, then reverse when nearly stopped
- Left arrow: steer left
- Right arrow: steer right
- `V`: cycle camera views
- `M`: mute or unmute audio

Things to notice while driving:

- The track has walls. Hitting them damages the car.
- The damage bar fills from `0.0` to `1.0`. At `1.0`, the car is fully damaged.
- Different camera views make different problems easier to see. Press `V` a
  few times while driving.
- Sound can be muted in the window with `M`.

You can also start with audio muted:

```bash
uv run racing --muted
```


## 4. Run The crash_fast Controller Demo

Open the starter controller:

```bash
src/controllers/crash_fast.py
```

It is intentionally tiny:

```python
from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Crash Fast"


def control(sensors: RobotSensors) -> RobotCommand:
    throttle: float = 1.0
    steer: float = 0.0
    brake: float = 0.0
    return RobotCommand(throttle, steer, brake)
```

This controller always holds full throttle, never steers, and never brakes. It
is useful because it proves that a controller can load and because its behavior
is easy to understand.

Run only `crash_fast` while watching it in the viewer:

```bash
uv run racing --student-module controllers.crash_fast
```

Watch what happens to the damage bar when the car reaches a wall. This is the
first thing your controller will try to improve.

## 5. Create Your Own Controller

Copy `crash_fast.py` to a new file in the controllers package:

```bash
cp src/controllers/crash_fast.py src/controllers/avoid_walls.py
```

Open `src/controllers/avoid_walls.py` and rename the racer:

```python
from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Avoid Walls"
RACING_COLOR: str = "#1EB4FF"


def control(sensors: RobotSensors) -> RobotCommand:
    throttle: float = 0.4
    steer: float = 0.0
    brake: float = 0.0
    return RobotCommand(throttle=throttle, steer=steer, brake=brake)
```

Run your controller:

```bash
uv run racing --student-module controllers.avoid_walls
```

The simulator calls your `control` function 60 times per second. Each call
receives the latest `RobotSensors` readings and must return a `RobotCommand`.

Command values are normalized:

- `throttle`: `-1.0` reverse to `1.0` forward
- `steer`: `-1.0` left to `1.0` right
- `brake`: `0.0` no brake to `1.0` full brake

## 6. First Sensor Strategy

Start with considering what inputs you might find valuable from any of the following three sensor groups:

- `sensors.wall_lidar`: distances to track walls
- `sensors.odometry.speed_mps`: current forward speed in meters per second
- `sensors.camera.center_offset_m` and
  `sensors.camera.heading_error_degrees`: where the track center is and which
  direction the track is going

Useful wall readings:

- `sensors.wall_lidar.front_m`: wall distance straight ahead
- `sensors.wall_lidar.front_left_m`: wall distance ahead-left
- `sensors.wall_lidar.front_right_m`: wall distance ahead-right
- `sensors.wall_lidar.left_m`: wall distance to the left
- `sensors.wall_lidar.right_m`: wall distance to the right

Useful camera readings:

- `sensors.camera.center_offset_m`: positive means the track center is to your
  right; negative means it is to your left
- `sensors.camera.heading_error_degrees`: positive means the track direction
  turns to your right; negative means it turns to your left

A reasonable first goal is:

- If you are too far from the centerline, steer back toward it.
- If you are too close to the left wall, steer right.
- If you are too close to the right wall, steer left.
- If a wall is close in front, brake and steer toward the more open side.
- Use `sensors.odometry.speed_mps` to avoid picking up too much speed.

Tune small pieces at a time. Steering, throttle, and brake interact with each
other, so a controller that turns well at low speed might crash at high speed.

## 7. Compare Racing Strategies

Once your controller can survive part of the track, start comparing it against
other drivers. Comparisons help you see whether a change actually improves the
race, not just whether it feels better in one test drive.

You can race against your controller with the keyboard:

```bash
uv run racing h2h --watch \
  --challenger-keyboard \
  --incumbent-student-module controllers.avoid_walls \
  --races 1 \
  --round-seconds 60 \
  --camera follow
```

Keyboard head-to-head races require `--watch` because you need the viewer to
drive. This is a good way to understand what your controller is struggling with:
try driving the same section yourself, then think about which sensor readings
could help your automated driver make a similar decision.

You can also watch your controller race an automated driver such as
`crash_fast`:

```bash
uv run racing h2h --watch \
  --challenger-student-module controllers.avoid_walls \
  --incumbent-student-module controllers.crash_fast \
  --races 1 \
  --round-seconds 60
```

The `--watch` flag opens the graphical viewer. Without `--watch`, head-to-head
can run without opening a window and print race results in the terminal. This is
called a headless race, and it is usually faster because the computer does not
have to render graphics or play sound.

Headless races need automated drivers on both sides:

```bash
uv run racing h2h \
  --challenger-student-module controllers.avoid_walls \
  --incumbent-student-module controllers.crash_fast \
  --races 5 \
  --round-seconds 60
```

Use watched races when you need to see what happened. Use headless races when
you want faster results across several races.

A good workflow is to keep versions to make copies of current working controllers and start iterating on a new controller to compare with your old one to ensure you are making forward progress:

```bash
cp src/controllers/avoid_walls.py src/controllers/race_faster.py
```

Improve `src/controllers/race_faster.py`, then race your own versions
against each other:

```bash
uv run racing h2h --watch \
  --challenger-student-module controllers.race_faster \
  --incumbent-student-module controllers.avoid_walls \
  --races 1 \
  --round-seconds 60 \
  --camera follow
```

This makes progress visible. A change that feels better while watching one car
might lose distance or take more damage in a race.

## 8. Complete Student Sensor Guide

Your controller receives one `RobotSensors` object each tick:

```python
def control(sensors: RobotSensors) -> RobotCommand:
    ...
```

### `sensors.imu`

Orientation and acceleration readings:

- `sensors.imu.heading_degrees`: compass-like yaw heading. `0.0` points along
  world +Z; positive turns toward the robot's right.
- `sensors.imu.yaw_rate_degrees_per_s`: turn rate. Positive means rotating to
  the right.
- `sensors.imu.pitch_degrees`: nose-up or nose-down tilt.
- `sensors.imu.roll_degrees`: side-to-side tilt.
- `sensors.imu.forward_acceleration_mps2`: acceleration along the robot's
  forward axis. Positive means speeding up forward.
- `sensors.imu.lateral_acceleration_mps2`: sideways acceleration. Positive
  means acceleration toward the robot's right.

### `sensors.odometry`

Wheel/vehicle motion estimates:

- `sensors.odometry.speed_mps`: signed forward speed in meters per second.
  Positive means forward, negative means reversing.
- `sensors.odometry.distance_m`: accumulated distance traveled since the
  controller run began. This is odometry distance, not official lap progress.

### `sensors.lidar`

Distance readings for nearby barriers, robots, and blockers:

- `sensors.lidar.angles_degrees`: beam angles relative to the robot.
- `sensors.lidar.distances_m`: distance for each beam, aligned with
  `angles_degrees`.
- `sensors.lidar.max_distance_m`: maximum beam range.
- `sensors.lidar.front_m`: nearest reading near straight ahead.
- `sensors.lidar.front_left_m`: nearest shallow front-left reading.
- `sensors.lidar.front_right_m`: nearest shallow front-right reading.
- `sensors.lidar.left_m`: nearest left reading.
- `sensors.lidar.right_m`: nearest right reading.
- `sensors.lidar.distance_at_angle_degrees(angle)`: reading whose beam angle is
  closest to `angle`.

Beam angles use this convention:

- `0.0`: straight ahead
- Negative angles: left
- Positive angles: right

### `sensors.wall_lidar`

The same shape as `sensors.lidar`, but it detects track barriers only.

Use this when you want wall avoidance that ignores other cars. The most common
fields are:

- `sensors.wall_lidar.front_m`
- `sensors.wall_lidar.front_left_m`
- `sensors.wall_lidar.front_right_m`
- `sensors.wall_lidar.left_m`
- `sensors.wall_lidar.right_m`
- `sensors.wall_lidar.distance_at_angle_degrees(angle)`

### `sensors.camera`

Processed track-vision readings. These are not raw camera pixels; they are
already simplified into numbers your controller can use.

- `sensors.camera.visible`: whether track-center readings are available.
- `sensors.camera.center_offset_m`: sideways offset from the robot to the track
  center. Positive means the center is to the robot's right; negative means it
  is to the left.
- `sensors.camera.heading_error_degrees`: signed angle from the robot's current
  heading to the track direction. Positive means the desired direction turns to
  the robot's right.
- `sensors.camera.lookahead_offsets_m`: offsets to future track-center points.
  Each value is in meters, positive to the robot's right.
- `sensors.camera.lookahead_distances_m`: forward distances corresponding to
  `lookahead_offsets_m`.
- `sensors.camera.competitors`: up to the three closest opponent cars in
  head-to-head races.

Each competitor reading in `sensors.camera.competitors` has:

- `distance_m`: distance from your robot to the competitor.
- `angle_degrees`: signed angle from your robot's forward direction to the
  competitor. Negative is left, positive is right.
- `relative_heading_degrees`: signed angle from your robot's heading to the
  competitor's heading.
- `speed_mps`: competitor forward speed.
- `closing_speed_mps`: your forward speed minus the competitor's forward speed.
  Positive means you are gaining.

### `sensors.contact`

Collision and damage readings:

- `sensors.contact.wall`: seconds of continuous current contact with a track
  barrier.
- `sensors.contact.robot`: seconds of continuous current contact with another
  robot or blocker.
- `sensors.contact.any_contact`: seconds of continuous current contact with any
  object.
- `sensors.contact.damage`: accumulated vehicle damage from `0.0` to `1.0`.
  At `1.0`, the robot is fully damaged.
