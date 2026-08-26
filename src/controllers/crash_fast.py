"""Self-driving robotic race car controller demo."""

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Crash Fast"
RACING_COLOR: str = "#ff8700"


def control(sensors: RobotSensors) -> RobotCommand:
    throttle: float = 0.0
    steer: float = 0.0
    return RobotCommand(throttle, steer)
