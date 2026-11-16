"""Pluggable robot hardware adapters for FYP Mission Control."""

from adapters.base import Pose2D, RGBDFrame, RobotAdapter, VRAMStatus, VelocityCommand
from adapters.mock_robot import MockRobotAdapter
from adapters.xlerobot_real import XLeRobotRealAdapter

__all__ = [
    "Pose2D",
    "RGBDFrame",
    "RobotAdapter",
    "VRAMStatus",
    "VelocityCommand",
    "MockRobotAdapter",
    "XLeRobotRealAdapter",
]
