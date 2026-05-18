from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .controller import DroneCommand
from .waypoints import Waypoint


class FlightAdapter(Protocol):
    def connect(self) -> None: ...
    def set_takeoff_target_cm(self, target_cm: float | None) -> None: ...
    def takeoff(self) -> None: ...
    def send_command(self, command: DroneCommand, duration_s: float) -> None: ...
    def get_height_cm(self) -> float | None: ...
    def get_telemetry(self) -> dict[str, float | None]: ...
    def fly_waypoint(self, waypoint: Waypoint, tolerance_m: float, timeout_s: float) -> dict[str, Any]: ...
    def land(self) -> None: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class NullFlightAdapter:
    command_history: list[dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    airborne: bool = False
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def connect(self) -> None:
        self.connected = True

    def set_takeoff_target_cm(self, target_cm: float | None) -> None:
        self.command_history.append({"event": "set_takeoff_target_cm", "target_cm": target_cm})

    def takeoff(self) -> None:
        self.airborne = True
        self.command_history.append({"event": "takeoff"})

    def send_command(self, command: DroneCommand, duration_s: float) -> None:
        self.command_history.append({"command": command.as_dict(), "duration_s": round(float(duration_s), 3)})
        if command.land:
            self.airborne = False

    def get_height_cm(self) -> float | None:
        return None

    def get_telemetry(self) -> dict[str, float | None]:
        return {"pos_x_cm": None, "pos_y_cm": None, "yaw_deg": None, "height_cm": None}

    def fly_waypoint(self, waypoint: Waypoint, tolerance_m: float, timeout_s: float) -> dict[str, Any]:
        self.position_m = (float(waypoint.x_m), float(waypoint.y_m), float(waypoint.z_m))
        payload = {
            "event": "waypoint",
            "waypoint": waypoint.to_dict(),
            "tolerance_m": round(float(tolerance_m), 3),
            "timeout_s": round(float(timeout_s), 3),
            "position_m": {
                "x": round(self.position_m[0], 3),
                "y": round(self.position_m[1], 3),
                "z": round(self.position_m[2], 3),
            },
            "reached": True,
        }
        self.command_history.append(payload)
        return payload

    def land(self) -> None:
        self.airborne = False
        self.command_history.append({"event": "land"})

    def close(self) -> None:
        self.connected = False


@dataclass(slots=True)
class CoDroneEDUAdapter:
    command_pause_s: float = 0.02
    position_poll_s: float = 0.1
    takeoff_target_cm: float | None = None
    descend_timeout_s: float = 4.0
    _drone: Any | None = None
    _airborne: bool = False

    def set_takeoff_target_cm(self, target_cm: float | None) -> None:
        # Floor at 15 cm: below that ground effect makes the CoDrone unstable.
        self.takeoff_target_cm = None if target_cm is None else max(float(target_cm), 15.0)

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
        # The firmware auto-takeoff always climbs to a fixed (~80 cm) hover;
        # there is no library parameter to shorten it. Immediately descend to
        # the requested low target so the drone spends almost no time high.
        self._drone.takeoff()
        self._airborne = True
        time.sleep(0.5)
        if self.takeoff_target_cm is not None:
            self._descend_to_target(self.takeoff_target_cm)

    def _descend_to_target(self, target_cm: float) -> None:
        # Measure the firmware hover height, then drop to target using the
        # drone's onboard position controller (optical-flow + IMU + range).
        try:
            height = float(self._drone.get_height("cm"))
        except Exception:
            height = 0.0
        if 0.0 < height < 900.0:
            drop_m = max((height - target_cm) / 100.0, 0.0)
            if drop_m > 0.03:
                try:
                    # x, y, z(m, down=-), velocity(m/s), heading, rotVel
                    self._drone.send_absolute_position(0.0, 0.0, -drop_m, 0.5, 0, 0)
                    time.sleep(0.3)
                    return
                except Exception:
                    pass

        deadline = time.perf_counter() + max(float(self.descend_timeout_s), 0.5)
        while time.perf_counter() < deadline:
            try:
                height = float(self._drone.get_height("cm"))
            except Exception:
                break
            if height <= 0 or height >= 900 or height <= target_cm + 3.0:
                break
            error_cm = height - target_cm
            power = -int(min(max(error_cm, 5.0), 40.0))
            self._drone.set_throttle(power)
            self._drone.move(0.2)
        self._drone.set_throttle(0)
        self._drone.reset_move_values()
        time.sleep(0.3)

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

    def get_telemetry(self) -> dict[str, float | None]:
        self._require_drone()

        def _safe(call) -> float | None:
            try:
                value = float(call())
            except Exception:
                return None
            return value

        return {
            "pos_x_cm": _safe(lambda: self._drone.get_pos_x("cm")),
            "pos_y_cm": _safe(lambda: self._drone.get_pos_y("cm")),
            "yaw_deg": _safe(self._drone.get_angle_z),
            "height_cm": self.get_height_cm(),
        }

    def fly_waypoint(self, waypoint: Waypoint, tolerance_m: float, timeout_s: float) -> dict[str, Any]:
        self._require_drone()
        if not self._airborne:
            return {"reached": False, "reason": "not_airborne"}

        self._drone.send_absolute_position(
            float(waypoint.x_m),
            float(waypoint.y_m),
            float(waypoint.z_m),
            float(waypoint.speed_m_s),
            int(waypoint.heading_deg),
            int(waypoint.rotational_velocity_dps),
        )

        deadline = time.perf_counter() + max(float(timeout_s), 0.1)
        last_position = self._position_data()
        while time.perf_counter() < deadline:
            position = self._position_data()
            if position is not None:
                last_position = position
                distance = _distance_m(
                    (float(waypoint.x_m), float(waypoint.y_m), float(waypoint.z_m)),
                    position,
                )
                if distance <= float(tolerance_m):
                    if waypoint.hold_s > 0:
                        time.sleep(float(waypoint.hold_s))
                    return {
                        "reached": True,
                        "distance_m": round(distance, 3),
                        "position_m": _position_payload(position),
                    }
            time.sleep(self.position_poll_s)

        return {
            "reached": False,
            "reason": "timeout",
            "position_m": None if last_position is None else _position_payload(last_position),
        }

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

    def _position_data(self) -> tuple[float, float, float] | None:
        self._require_drone()
        try:
            payload = self._drone.get_position_data()
        except Exception:
            return None
        return _parse_position_data(payload)


def _parse_position_data(payload: Any) -> tuple[float, float, float] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        keys = ("x", "y", "z")
        if all(key in payload for key in keys):
            try:
                return float(payload["x"]), float(payload["y"]), float(payload["z"])
            except (TypeError, ValueError):
                return None
    if isinstance(payload, (list, tuple)) and len(payload) >= 3:
        try:
            return float(payload[0]), float(payload[1]), float(payload[2])
        except (TypeError, ValueError):
            return None
    for attribute_set in (("x", "y", "z"), ("X", "Y", "Z")):
        if all(hasattr(payload, attribute) for attribute in attribute_set):
            try:
                return tuple(float(getattr(payload, attribute)) for attribute in attribute_set)
            except (TypeError, ValueError):
                return None
    return None


def _position_payload(position_m: tuple[float, float, float]) -> dict[str, float]:
    return {
        "x": round(float(position_m[0]), 3),
        "y": round(float(position_m[1]), 3),
        "z": round(float(position_m[2]), 3),
    }


def _distance_m(target: tuple[float, float, float], actual: tuple[float, float, float]) -> float:
    return (
        ((float(target[0]) - float(actual[0])) ** 2)
        + ((float(target[1]) - float(actual[1])) ** 2)
        + ((float(target[2]) - float(actual[2])) ** 2)
    ) ** 0.5
