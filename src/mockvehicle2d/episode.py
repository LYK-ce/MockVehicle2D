"""Headless, fixed-tick fleet episodes for repeatable algorithm experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math

from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    CoverageMission,
    GotoMission,
    Mission,
    ModeAction,
    ModeCommand,
    PatrolMission,
)
from mockvehicle2d.fleet import FleetRuntime, FleetScenario
from mockvehicle2d.local_state import OdometryConfig
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.vehicle import (
    DEFAULT_ANGULAR_ACCELERATION_RPS2,
    DEFAULT_LINEAR_ACCELERATION_MPS2,
    DEFAULT_LINEAR_DECELERATION_MPS2,
)


RESULT_SCHEMA_VERSION = 2
MIN_PROGRESS_TRANSLATION_M = 0.001
PEER_STATE_DELAY_TICKS = 1
_MISSION_TYPES = (GotoMission, PatrolMission, CoverageMission)


@dataclass(frozen=True)
class EpisodeResult:
    scenario_id: str
    odometry_seed: int
    tick_count: int
    simulation_duration_s: float
    success: bool
    termination_reason: str
    minimum_inter_vehicle_clearance_m: float | None
    vehicles: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "determinism": {
                "clock": "fixed_tick",
                "odometry_seed": self.odometry_seed,
            },
            "tick_count": self.tick_count,
            "simulation_duration_s": self.simulation_duration_s,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "minimum_inter_vehicle_clearance_m": self.minimum_inter_vehicle_clearance_m,
            "vehicles": list(self.vehicles),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class _DeterministicPeerStateExchange:
    """Relay validated peer-state JSON after a fixed simulation-tick delay."""

    def __init__(self, fleet: FleetRuntime) -> None:
        self._fleet = fleet
        self._tick = 0
        self._pending: list[tuple[int, str, str]] = []
        vehicle_ids = tuple(sorted(fleet.nodes))
        peer_ids = {
            vehicle_id: f"episode-peer-{index}"
            for index, vehicle_id in enumerate(vehicle_ids, 1)
        }
        for vehicle_id in vehicle_ids:
            state = fleet.nodes[vehicle_id].map_sync
            assert state is not None
            state.configure_network(
                peer_ids[vehicle_id],
                {
                    other_id: (
                        peer_ids[other_id],
                        fleet.nodes[other_id].local_state.anchor,
                    )
                    for other_id in vehicle_ids
                    if other_id != vehicle_id
                },
            )
            state.set_health(
                ready=True,
                connected_vehicle_ids=(
                    other_id for other_id in vehicle_ids if other_id != vehicle_id
                ),
            )
        self._publish()

    def advance(self) -> None:
        self._tick += 1
        due = [item for item in self._pending if item[0] <= self._tick]
        self._pending = [item for item in self._pending if item[0] > self._tick]
        for _, source_id, encoded in due:
            source_peer_id = self._fleet.nodes[source_id].map_sync.local_peer_id
            assert source_peer_id is not None
            for receiver_id in sorted(self._fleet.nodes):
                if receiver_id == source_id:
                    continue
                receiver = self._fleet.nodes[receiver_id].map_sync
                assert receiver is not None
                receiver.receive_transport(
                    source_peer_id,
                    source_id,
                    json.loads(encoded),
                )
        self._publish()

    def _publish(self) -> None:
        if len(self._fleet.nodes) < 2:
            return
        for source_id in sorted(self._fleet.nodes):
            state = self._fleet.nodes[source_id].map_sync
            assert state is not None
            payload = state.prepare_peer_state()
            if payload is None:
                continue
            encoded = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._pending.append(
                (self._tick + PEER_STATE_DELAY_TICKS, source_id, encoded)
            )
            state.publish_transport_result(
                str(payload["protocol"]),
                int(payload["sequence"]),
                True,
            )


def run_episode(
    scenario: FleetScenario,
    missions_by_vehicle: Mapping[str, Sequence[Mission]],
    *,
    max_simulation_s: float,
    grid: MapGrid | None = None,
    voxels: list[dict[str, object]] | None = None,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    linear_acceleration_mps2: float = DEFAULT_LINEAR_ACCELERATION_MPS2,
    linear_deceleration_mps2: float = DEFAULT_LINEAR_DECELERATION_MPS2,
    angular_acceleration_rps2: float = DEFAULT_ANGULAR_ACCELERATION_RPS2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
    mission_capacity: int = 16,
    odometry_config: OdometryConfig = OdometryConfig(),
    realtime_factor: float = 1.0,
) -> EpisodeResult:
    """Run initial missions synchronously until completion, blockage, or timeout."""
    if scenario.p2p is not None:
        raise ValueError(
            "Episode Runner requires p2p to be disabled; deterministic "
            "communication is not implemented"
        )
    if (
        isinstance(max_simulation_s, bool)
        or not isinstance(max_simulation_s, (int, float))
        or not math.isfinite(max_simulation_s)
        or max_simulation_s <= 0
    ):
        raise ValueError("max_simulation_s must be finite and positive")
    if not isinstance(missions_by_vehicle, Mapping):
        raise ValueError("missions_by_vehicle must be a mapping")
    if any(not isinstance(vehicle_id, str) for vehicle_id in missions_by_vehicle):
        raise ValueError("mission vehicle ids must be strings")

    known_vehicle_ids = {spec.vehicle_id for spec in scenario.vehicles}
    unknown_vehicle_ids = set(missions_by_vehicle) - known_vehicle_ids
    if unknown_vehicle_ids:
        raise ValueError(
            "missions reference unknown vehicles: "
            + ", ".join(sorted(unknown_vehicle_ids))
        )

    missions: dict[str, tuple[Mission, ...]] = {}
    for vehicle_id in sorted(known_vehicle_ids):
        vehicle_missions = missions_by_vehicle.get(vehicle_id, ())
        if isinstance(vehicle_missions, (str, bytes)) or not isinstance(
            vehicle_missions, Sequence
        ):
            raise ValueError(f"missions for {vehicle_id} must be a sequence")
        missions[vehicle_id] = tuple(vehicle_missions)
        if any(
            not isinstance(mission, _MISSION_TYPES)
            for mission in missions[vehicle_id]
        ):
            raise ValueError(f"missions for {vehicle_id} contain an unsupported value")
    if not any(missions.values()):
        raise ValueError("Episode Runner requires at least one mission")

    fleet = FleetRuntime.create(
        scenario,
        started_at=0.0,
        timestamp=0.0,
        grid=grid,
        voxels=voxels,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        linear_acceleration_mps2=linear_acceleration_mps2,
        linear_deceleration_mps2=linear_deceleration_mps2,
        angular_acceleration_rps2=angular_acceleration_rps2,
        radius=radius,
        command_timeout=command_timeout,
        mission_capacity=mission_capacity,
        odometry_config=odometry_config,
        realtime_factor=realtime_factor,
        in_process_peer_states=len(scenario.vehicles) > 1,
    )
    peer_exchange = (
        None
        if len(scenario.vehicles) < 2
        else _DeterministicPeerStateExchange(fleet)
    )
    for vehicle_id in sorted(missions):
        if not missions[vehicle_id]:
            continue
        mode_result = fleet.handle_command(
            vehicle_id,
            ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
        )
        push_result = fleet.handle_command(
            vehicle_id,
            AutoCommand(2, AutoAction.PUSH, missions[vehicle_id]),
        )
        if not mode_result.accepted or not push_result.accepted:
            reason = mode_result.reason or push_result.reason or "command_rejected"
            raise ValueError(f"cannot start missions for {vehicle_id}: {reason}")

    previous_poses = fleet.world.truth_snapshot()
    radii = {
        vehicle_id: fleet.world.vehicle(vehicle_id).radius
        for vehicle_id in previous_poses
    }
    path_lengths = {vehicle_id: 0.0 for vehicle_id in previous_poses}
    no_progress_ticks = {vehicle_id: 0 for vehicle_id in previous_poses}
    longest_no_progress_ticks = {vehicle_id: 0 for vehicle_id in previous_poses}
    minimum_clearance = _minimum_inter_vehicle_clearance(previous_poses, radii)
    collisions = {vehicle_id: False for vehicle_id in previous_poses}
    safety_stops = {vehicle_id: False for vehicle_id in previous_poses}
    tick_count = 0
    termination_reason: str | None = None
    terminal_statuses: dict[tuple[str, str], str] | None = None

    while True:
        statuses = _mission_statuses(fleet, missions)
        if termination_reason is None and all(
            status == "reached" for status in statuses.values()
        ):
            termination_reason = "completed"
            terminal_statuses = statuses
        elif termination_reason is None and any(
            status == "blocked" for status in statuses.values()
        ):
            termination_reason = "blocked"
            terminal_statuses = statuses
            for vehicle_id in sorted(missions):
                if (
                    missions[vehicle_id]
                    and fleet.nodes[vehicle_id].controller.auto_state.value == "active"
                ):
                    fleet.handle_command(
                        vehicle_id,
                        ModeCommand(3, ModeAction.STOP_MOTION),
                    )
        if termination_reason is not None and _vehicles_stopped(
            fleet,
            tuple(
                vehicle_id
                for vehicle_id, vehicle_missions in missions.items()
                if vehicle_missions
            ),
        ):
            break
        if (tick_count + 1) * fleet.tick_s > max_simulation_s + 1e-12:
            termination_reason = "timeout"
            break

        fleet.tick(fleet.timestamp_at(fleet.world.now + fleet.tick_s))
        tick_count += 1
        if peer_exchange is not None:
            peer_exchange.advance()
        current_poses = fleet.world.truth_snapshot()
        for vehicle_id, pose in current_poses.items():
            previous = previous_poses[vehicle_id]
            translation_m = math.hypot(
                pose[0] - previous[0],
                pose[1] - previous[1],
            )
            path_lengths[vehicle_id] += translation_m
            (
                no_progress_ticks[vehicle_id],
                longest_no_progress_ticks[vehicle_id],
            ) = _update_no_progress(
                no_progress_ticks[vehicle_id],
                longest_no_progress_ticks[vehicle_id],
                translation_m,
            )
            vehicle = fleet.world.vehicle(vehicle_id)
            node = fleet.nodes[vehicle_id]
            collisions[vehicle_id] |= vehicle.collision
            safety_stops[vehicle_id] |= node.safety.decision.state in {
                "stopped",
                "fault",
            }
        current_clearance = _minimum_inter_vehicle_clearance(current_poses, radii)
        if current_clearance is not None:
            assert minimum_clearance is not None
            minimum_clearance = min(minimum_clearance, current_clearance)
        previous_poses = current_poses

    assert termination_reason is not None
    statuses = terminal_statuses or _mission_statuses(fleet, missions)
    vehicle_results = []
    for vehicle_id in sorted(fleet.nodes):
        node = fleet.nodes[vehicle_id]
        vehicle = fleet.world.vehicle(vehicle_id)
        blocked_reason = (
            node.controller.navigation.reason
            if node.controller.auto_state.value == "blocked"
            else None
        )
        safety_stops[vehicle_id] |= bool(
            blocked_reason and blocked_reason.startswith("safety_")
        )
        vehicle_results.append(
            {
                "vehicle_id": vehicle_id,
                "final_pose": {
                    "source": "simulator_ground_truth",
                    "x_m": _stable_float(vehicle.x),
                    "y_m": _stable_float(vehicle.y),
                    "yaw_rad": _stable_float(vehicle.yaw),
                },
                "path_length_m": _stable_float(path_lengths[vehicle_id]),
                "longest_no_progress_duration_s": _stable_float(
                    longest_no_progress_ticks[vehicle_id] * fleet.tick_s
                ),
                "collision_occurred": collisions[vehicle_id],
                "blocked": node.controller.auto_state.value == "blocked",
                "blocked_reason": blocked_reason,
                "safety_stop_occurred": safety_stops[vehicle_id],
                "final_safety": {
                    "state": node.safety.decision.state,
                    "reason": node.safety.decision.reason,
                },
                "missions": [
                    {
                        "mission_id": mission.mission_id,
                        "type": mission.mission_type,
                        "status": statuses[(vehicle_id, mission.mission_id)],
                    }
                    for mission in missions[vehicle_id]
                ],
            }
        )

    return EpisodeResult(
        scenario.scenario_id,
        odometry_config.seed,
        tick_count,
        _stable_float(tick_count * fleet.tick_s),
        termination_reason == "completed",
        termination_reason,
        None if minimum_clearance is None else _stable_float(minimum_clearance),
        tuple(vehicle_results),
    )


def _mission_statuses(
    fleet: FleetRuntime,
    missions: Mapping[str, Sequence[Mission]],
) -> dict[tuple[str, str], str]:
    statuses = {
        (vehicle_id, mission.mission_id): "not_started"
        for vehicle_id, vehicle_missions in missions.items()
        for mission in vehicle_missions
    }
    for vehicle_id in sorted(missions):
        for event in fleet.nodes[vehicle_id].controller.events_after(0):
            key = vehicle_id, event.mission.mission_id
            if key in statuses:
                statuses[key] = event.status
    return statuses


def _vehicles_stopped(
    fleet: FleetRuntime,
    vehicle_ids: Sequence[str],
) -> bool:
    return all(
        fleet.world.vehicle(vehicle_id).target_velocities() == (0.0, 0.0)
        and fleet.world.vehicle(vehicle_id).body_velocities() == (0.0, 0.0)
        for vehicle_id in vehicle_ids
    )


def _stable_float(value: float) -> float:
    return round(float(value), 12)


def _minimum_inter_vehicle_clearance(
    poses: Mapping[str, tuple[float, float, float]],
    radii: Mapping[str, float],
) -> float | None:
    vehicle_ids = sorted(poses)
    if len(vehicle_ids) < 2:
        return None
    return min(
        math.dist(poses[first][:2], poses[second][:2])
        - radii[first]
        - radii[second]
        for index, first in enumerate(vehicle_ids)
        for second in vehicle_ids[index + 1 :]
    )


def _update_no_progress(
    current_ticks: int,
    longest_ticks: int,
    translation_m: float,
) -> tuple[int, int]:
    if translation_m >= MIN_PROGRESS_TRANSLATION_M:
        return 0, longest_ticks
    current_ticks += 1
    return current_ticks, max(longest_ticks, current_ticks)
