"""Pure safety observations and velocity limiting for the 2D simulator."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.scan import LaserPoint, TMINI_SCAN_CONFIG, scan_grid
from mockvehicle2d.vehicle import Vehicle


HARD_STOP_CLEARANCE_M = 0.25
SLOW_ZONE_CLEARANCE_M = 1.0
OBSTACLE_SECTOR_HALF_ANGLE_RAD = math.radians(30)
EDGE_LOOKAHEAD_M = 2.0
EDGE_SAMPLE_STEP_M = 0.05


@dataclass(frozen=True)
class SafetyObservation:
    obstacle_clearance_m: float | None = None
    edge_clearance_m: float | None = None
    healthy: bool = True

    def __post_init__(self) -> None:
        for clearance in (self.obstacle_clearance_m, self.edge_clearance_m):
            if clearance is not None and (not math.isfinite(clearance) or clearance < 0):
                raise ValueError("safety clearances must be finite and non-negative")
        if type(self.healthy) is not bool:
            raise ValueError("healthy must be a bool")


@dataclass(frozen=True)
class SafetyDecision:
    linear_mps: float
    angular_rps: float
    state: str
    reason: str | None


def nearest_obstacle_clearance(
    points: Iterable[LaserPoint],
    desired_linear_mps: float,
    vehicle_radius: float,
    sector_half_angle_rad: float = OBSTACLE_SECTOR_HALF_ANGLE_RAD,
) -> float | None:
    """Nearest positive Tmini return in the forward or reverse travel sector."""
    if desired_linear_mps == 0:
        return None
    if not math.isfinite(desired_linear_mps) or not math.isfinite(vehicle_radius) or vehicle_radius < 0:
        raise ValueError("motion and radius must be finite; radius cannot be negative")
    if not math.isfinite(sector_half_angle_rad) or not 0 < sector_half_angle_rad <= math.pi:
        raise ValueError("sector half angle must be in (0, pi]")

    center = 0.0 if desired_linear_mps > 0 else math.pi
    ranges = (
        point.range
        for point in points
        if math.isfinite(point.angle)
        and math.isfinite(point.range)
        and point.range > 0
        and abs(math.atan2(math.sin(point.angle - center), math.cos(point.angle - center)))
        <= sector_half_angle_rad
    )
    nearest = min(ranges, default=None)
    return None if nearest is None else max(0.0, nearest - vehicle_radius)


def nearest_edge_clearance(
    grid: MapGrid,
    x: float,
    y: float,
    yaw: float,
    desired_linear_mps: float,
    *,
    vehicle_radius: float,
    lookahead_m: float = EDGE_LOOKAHEAD_M,
    sample_step_m: float = EDGE_SAMPLE_STEP_M,
) -> float | None:
    """Conservatively sample the swept circular footprint for void or map bounds."""
    values = (x, y, yaw, desired_linear_mps, vehicle_radius, lookahead_m, sample_step_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("edge sensing inputs must be finite")
    if vehicle_radius <= 0 or lookahead_m < 0 or sample_step_m <= 0:
        raise ValueError("radius and sample step must be positive; lookahead cannot be negative")
    if desired_linear_mps == 0:
        return None

    direction = yaw if desired_linear_mps > 0 else yaw + math.pi
    sample_count = math.ceil(lookahead_m / sample_step_m)
    for index in range(sample_count + 1):
        distance = min(index * sample_step_m, lookahead_m)
        center_x = x + distance * math.cos(direction)
        center_y = y + distance * math.sin(direction)
        if not _footprint_has_ground(grid, center_x, center_y, vehicle_radius):
            return max(0.0, distance - sample_step_m) if index else 0.0
    return None


def _footprint_has_ground(grid: MapGrid, cx: float, cy: float, radius: float) -> bool:
    radius_squared = radius * radius
    for gy in range(math.floor(cy - radius), math.floor(cy + radius) + 1):
        for gx in range(math.floor(cx - radius), math.floor(cx + radius) + 1):
            closest_x = max(gx, min(cx, gx + 1))
            closest_y = max(gy, min(cy, gy + 1))
            if (closest_x - cx) ** 2 + (closest_y - cy) ** 2 < radius_squared:
                if not grid.has_ground(gx, gy):
                    return False
    return True


class SafetyGovernor:
    """Apply fixed safety policy without owning sensors or vehicle state."""

    def limit(
        self,
        desired_linear_mps: float,
        desired_angular_rps: float,
        observation: SafetyObservation,
        automatic: bool,
    ) -> SafetyDecision:
        values = (desired_linear_mps, desired_angular_rps)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("desired velocities must be numbers")
        if not all(math.isfinite(value) for value in values) or type(automatic) is not bool:
            raise ValueError("desired velocities must be finite and automatic must be a bool")

        linear_mps = float(desired_linear_mps)
        angular_rps = float(desired_angular_rps)
        if not observation.healthy:
            return SafetyDecision(0.0, angular_rps, "fault", "safety_sensor_fault")

        nearest = min(
            (
                (clearance, reason)
                for clearance, reason in (
                    (observation.obstacle_clearance_m, "safety_obstacle"),
                    (observation.edge_clearance_m, "safety_edge"),
                )
                if clearance is not None
            ),
            default=None,
        )
        if nearest is None:
            return SafetyDecision(linear_mps, angular_rps, "clear", None)

        clearance, reason = nearest
        if clearance <= HARD_STOP_CLEARANCE_M:
            return SafetyDecision(0.0, angular_rps, "stopped", reason)
        if automatic and linear_mps and clearance < SLOW_ZONE_CLEARANCE_M:
            scale = (clearance - HARD_STOP_CLEARANCE_M) / (
                SLOW_ZONE_CLEARANCE_M - HARD_STOP_CLEARANCE_M
            )
            return SafetyDecision(linear_mps * scale, angular_rps, "limited", reason)
        return SafetyDecision(linear_mps, angular_rps, "clear", None)


class LocalSafetyRuntime:
    """Sample local safety inputs, apply policy, and retain observable state."""

    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.observation = SafetyObservation(healthy=healthy)
        self.decision = SafetyDecision(
            0.0,
            0.0,
            "clear" if healthy else "fault",
            None if healthy else "safety_sensor_fault",
        )
        self._governor = SafetyGovernor()

    def evaluate(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        desired_linear_mps: float,
        desired_angular_rps: float,
        *,
        automatic: bool,
    ) -> SafetyDecision:
        points = (
            scan_grid(grid, vehicle.x, vehicle.y, vehicle.yaw, TMINI_SCAN_CONFIG)
            if desired_linear_mps
            else ()
        )
        self.observation = SafetyObservation(
            obstacle_clearance_m=nearest_obstacle_clearance(
                points, desired_linear_mps, vehicle.radius
            ),
            edge_clearance_m=nearest_edge_clearance(
                grid,
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                desired_linear_mps,
                vehicle_radius=vehicle.radius,
            ),
            healthy=self.healthy,
        )
        self.decision = self._governor.limit(
            desired_linear_mps, desired_angular_rps, self.observation, automatic
        )
        return self.decision

    def enforce_manual(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        desired: tuple[float, float] | None = None,
    ) -> SafetyDecision:
        """Apply hard manual safety; a stopped latch is changed only by a new command."""
        velocities = vehicle.body_velocities() if desired is None else desired
        if desired is None and velocities == (0.0, 0.0):
            return self.decision
        decision = self.evaluate(vehicle, grid, *velocities, automatic=False)
        if decision.state in {"stopped", "fault"}:
            vehicle.stop()
        return decision

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.decision.state,
            "reason": self.decision.reason,
            "obstacle_clearance_m": self.observation.obstacle_clearance_m,
            "edge_clearance_m": self.observation.edge_clearance_m,
        }
