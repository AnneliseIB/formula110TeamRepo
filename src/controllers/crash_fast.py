"""Self-driving robotic race car controller demo."""

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Crash Fast"


def control(sensors: RobotSensors) -> RobotCommand:
    throttle: float = 1.0
    steer: float = 0.0
    brake: float = 0.0
    return RobotCommand(throttle, steer, brake)
