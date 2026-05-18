from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .controller import DroneCommand


class FlightAdapter(Protocol):
    def connect(self) -> None: ...
    def takeoff(self) -> None: ...
    def send_command(self, command: DroneCommand, duration_s: float) -> None: ...
    def get_height_cm(self) -> float | None: ...
    def land(self) -> None: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class NullFlightAdapter:
    command_history: list[dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    airborne: bool = False

    def connect(self) -> None:
        self.connected = True

    def takeoff(self) -> None:
        self.airborne = True
        self.command_history.append({"event": "takeoff"})

    def send_command(self, command: DroneCommand, duration_s: float) -> None:
        self.command_history.append({"command": command.as_dict(), "duration_s": round(float(duration_s), 3)})
        if command.land:
            self.airborne = False

    def get_height_cm(self) -> float | None:
        return None

    def land(self) -> None:
        self.airborne = False
        self.command_history.append({"event": "land"})

    def close(self) -> None:
        self.connected = False


@dataclass(slots=True)
class CoDroneEDUAdapter:
    command_pause_s: float = 0.02
    _drone: Any | None = None
    _airborne: bool = False

    def connect(self) -> None:
        try:
            from codrone_edu.drone import Drone
        except ImportError as exc:
            raise ImportError(
                "CoDrone EDU Python library is not installed. Install the official "
                "`codrone_edu` package before using follow-track."
            ) from exc

        self._drone = Drone()
        self._drone.pair()

    def takeoff(self) -> None:
        self._require_drone()
        self._drone.takeoff()
        self._airborne = True
        time.sleep(1.0)

    def send_command(self, command: DroneCommand, duration_s: float) -> None:
        self._require_drone()
        if not self._airborne:
            return
        if command.land:
            self.land()
            return

        self._drone.set_roll(int(command.roll))
        self._drone.set_pitch(int(command.pitch))
        self._drone.set_yaw(int(command.yaw))
        self._drone.set_throttle(int(command.throttle))
        self._drone.move(max(float(duration_s), 0.05))
        self._drone.reset_move_values()
        time.sleep(self.command_pause_s)

    def get_height_cm(self) -> float | None:
        self._require_drone()
        try:
            height = self._drone.get_height("cm")
        except Exception:
            return None
        try:
            height_value = float(height)
        except (TypeError, ValueError):
            return None
        if height_value <= 0 or height_value >= 900:
            return None
        return height_value

    def land(self) -> None:
        if self._drone is None:
            return
        if not self._airborne:
            return
        self._drone.land()
        self._airborne = False
        time.sleep(1.0)

    def close(self) -> None:
        if self._drone is None:
            return
        try:
            if self._airborne:
                self.land()
        finally:
            self._drone.close()
            self._drone = None
            self._airborne = False

    def _require_drone(self) -> None:
        if self._drone is None:
            raise RuntimeError("CoDrone adapter is not connected.")
