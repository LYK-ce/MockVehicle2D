"""Public Episode scenarios that expose coordination capability limits."""

import math
import unittest

import pytest

from mockvehicle2d.controller import GotoMission
from mockvehicle2d.episode import EpisodeResult, run_episode
from mockvehicle2d.fleet import AnchorPose, FleetScenario, FleetVehicleSpec
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import AUTOMATIC_MINIMUM_CLEARANCE_M


def _fork_grid() -> MapGrid:
    walls = {
        (x, y)
        for x in range(21)
        for y in (1, 13)
    } | {
        (x, y)
        for x in (0, 20)
        for y in range(1, 14)
    } | {
        (x, y)
        for x in range(8, 13)
        for y in range(5, 10)
    }
    return MapGrid.from_wall_set(21, 15, walls)


def _passing_bay_grid() -> MapGrid:
    walls = {
        (x, y)
        for x in range(25)
        for y in (0, 7, 10)
    } | {
        (x, y)
        for x in (0, 24)
        for y in range(11)
    } | {
        (x, 3)
        for x in range(1, 24)
        if not 10 <= x <= 14
    } | {
        (x, y)
        for x in (9, 15)
        for y in range(1, 4)
    }
    return MapGrid.from_wall_set(25, 11, walls)


def _nested_passing_bay_grid() -> MapGrid:
    walls = {
        (x, y)
        for x in range(25)
        for y in (0, 3, 17)
    } | {
        (x, 7)
        for x in range(25)
        if not 11 <= x <= 13
    } | {
        (x, y)
        for x in (0, 24)
        for y in range(18)
    } | {
        (x, y)
        for x in (10, 14)
        for y in range(7, 18)
    }
    return MapGrid.from_wall_set(25, 18, walls)


def _queued_t_junction_grid() -> MapGrid:
    free = {
        (x, y)
        for x in range(1, 22)
        for y in range(4, 9)
    } | {
        (x, y)
        for x in range(10, 13)
        for y in range(4, 22)
    }
    walls = {
        (x, y)
        for x in range(23)
        for y in range(23)
        if (x, y) not in free
    }
    return MapGrid.from_wall_set(23, 23, walls)


class TestCoordinationCapabilityGaps(unittest.TestCase):
    def assert_safe_success(self, result: EpisodeResult) -> None:
        self.assertTrue(
            result.success
            and all(not vehicle["collision_occurred"] for vehicle in result.vehicles),
            result.as_dict(),
        )

    def assert_campaign_success(
        self,
        case_id: str,
        result: EpisodeResult,
        *,
        max_no_progress_s: float,
    ) -> None:
        evidence = f"{case_id}: {result.to_json()}"
        self.assertTrue(result.success, evidence)
        self.assertEqual(result.termination_reason, "completed", evidence)
        if result.minimum_inter_vehicle_clearance_m is not None:
            self.assertGreaterEqual(
                result.minimum_inter_vehicle_clearance_m,
                AUTOMATIC_MINIMUM_CLEARANCE_M,
                evidence,
            )
        self.assertTrue(
            all(
                not vehicle["collision_occurred"]
                and not vehicle["blocked"]
                and all(
                    mission["status"] == "reached"
                    for mission in vehicle["missions"]
                )
                and vehicle["longest_no_progress_duration_s"]
                <= max_no_progress_s
                for vehicle in result.vehicles
            ),
            evidence,
        )

    @pytest.mark.extended
    def test_nested_four_vehicle_chain_has_a_staged_physical_solution(self) -> None:
        grid = _nested_passing_bay_grid()
        phases = (
            (
                "move-d",
                {"vehicle_d": (12.5, 12.5, math.pi / 2)},
                {"vehicle_d": (12.5, 15.5)},
            ),
            (
                "move-c",
                {
                    "vehicle_c": (12.5, 9.5, math.pi / 2),
                    "vehicle_d": (12.5, 15.5, math.pi / 2),
                },
                {
                    "vehicle_c": (12.5, 12.5),
                    "vehicle_d": (12.5, 15.5),
                },
            ),
            (
                "move-b",
                {
                    "vehicle_b": (20.5, 5.5, math.pi),
                    "vehicle_c": (12.5, 12.5, math.pi / 2),
                    "vehicle_d": (12.5, 15.5, math.pi / 2),
                },
                {
                    "vehicle_b": (12.5, 9.5),
                    "vehicle_c": (12.5, 12.5),
                    "vehicle_d": (12.5, 15.5),
                },
            ),
            (
                "cross-a",
                {
                    "vehicle_a": (4.5, 5.5, 0.0),
                    "vehicle_b": (12.5, 9.5, math.pi / 2),
                    "vehicle_c": (12.5, 12.5, math.pi / 2),
                    "vehicle_d": (12.5, 15.5, math.pi / 2),
                },
                {
                    "vehicle_a": (20.5, 5.5),
                    "vehicle_b": (12.5, 9.5),
                    "vehicle_c": (12.5, 12.5),
                    "vehicle_d": (12.5, 15.5),
                },
            ),
        )
        for case_id, starts, goals in phases:
            with self.subTest(case_id=case_id):
                result = run_episode(
                    FleetScenario(
                        f"nested_chain_oracle_{case_id}",
                        tuple(
                            FleetVehicleSpec(
                                vehicle_id,
                                19090 + ord(vehicle_id[-1]) - ord("a"),
                                f"{vehicle_id}_spawn",
                                AnchorPose(*pose),
                            )
                            for vehicle_id, pose in starts.items()
                        ),
                        100,
                    ),
                    {
                        vehicle_id: (
                            GotoMission(
                                f"{case_id}-{vehicle_id}",
                                "global_map",
                                *goal,
                                2,
                            ),
                        )
                        for vehicle_id, goal in goals.items()
                    },
                    max_simulation_s=60.0,
                    grid=grid,
                    linear_speed=1.0,
                )
                self.assert_campaign_success(
                    f"nested-chain-oracle-{case_id}",
                    result,
                    max_no_progress_s=30.0,
                )

    @pytest.mark.extended
    def test_queued_t_junction_has_a_staged_physical_solution(self) -> None:
        positions = {
            "vehicle_a": (4.5, 6.5, 0.0),
            "vehicle_b": (18.5, 6.5, math.pi),
            "vehicle_c": (11.5, 11.5, -math.pi / 2),
            "vehicle_d": (11.5, 16.5, -math.pi / 2),
        }
        steps = (
            ("stage-c", "vehicle_c", (8.5, 5.0)),
            ("stage-d", "vehicle_d", (14.5, 5.0)),
            ("move-b", "vehicle_b", (11.5, 16.5)),
            ("move-d", "vehicle_d", (11.5, 11.5)),
            ("move-a", "vehicle_a", (18.5, 6.5)),
            ("move-c", "vehicle_c", (4.5, 6.5)),
        )
        for case_id, moving_vehicle_id, destination in steps:
            result = run_episode(
                FleetScenario(
                    f"queued_t_junction_oracle_{case_id}",
                    tuple(
                        FleetVehicleSpec(
                            vehicle_id,
                            19090 + ord(vehicle_id[-1]) - ord("a"),
                            f"{vehicle_id}_spawn",
                            AnchorPose(*pose),
                        )
                        for vehicle_id, pose in positions.items()
                    ),
                    100,
                ),
                {
                    vehicle_id: (
                        GotoMission(
                            f"{case_id}-{vehicle_id}",
                            "global_map",
                            *(
                                destination
                                if vehicle_id == moving_vehicle_id
                                else positions[vehicle_id][:2]
                            ),
                            2,
                        ),
                    )
                    for vehicle_id in positions
                },
                max_simulation_s=60.0,
                grid=_queued_t_junction_grid(),
                linear_speed=1.0,
            )
            self.assert_campaign_success(
                f"queued-t-junction-oracle-{case_id}",
                result,
                max_no_progress_s=30.0,
            )
            positions[moving_vehicle_id] = (*destination, 0.0)

    @pytest.mark.extended
    def test_nested_four_vehicle_chain_completes_simultaneous_gotos(self) -> None:
        starts = {
            "vehicle_a": (4.5, 5.5, 0.0),
            "vehicle_b": (20.5, 5.5, math.pi),
            "vehicle_c": (12.5, 9.5, math.pi / 2),
            "vehicle_d": (12.5, 12.5, math.pi / 2),
        }
        goals = {
            "vehicle_a": (20.5, 5.5),
            "vehicle_b": (12.5, 9.5),
            "vehicle_c": (12.5, 12.5),
            "vehicle_d": (12.5, 15.5),
        }
        results = tuple(
            run_episode(
                FleetScenario(
                    "nested_chain_simultaneous_gotos",
                    tuple(
                        FleetVehicleSpec(
                            vehicle_id,
                            19090 + ord(vehicle_id[-1]) - ord("a"),
                            f"{vehicle_id}_spawn",
                            AnchorPose(*starts[vehicle_id]),
                        )
                        for vehicle_id in order
                    ),
                    100,
                ),
                {
                    vehicle_id: (
                        GotoMission(
                            f"simultaneous-{vehicle_id}",
                            "global_map",
                            *goal,
                            2,
                        ),
                    )
                    for vehicle_id, goal in goals.items()
                },
                max_simulation_s=90.0,
                grid=_nested_passing_bay_grid(),
                linear_speed=1.0,
            )
            for order in (tuple(starts), tuple(reversed(starts)))
        )
        for order, result in zip(
            ("canonical", "reverse-order"),
            results,
        ):
            self.assert_campaign_success(
                f"nested-chain-simultaneous-gotos-{order}",
                result,
                max_no_progress_s=60.0,
            )
        self.assertEqual(results[0].to_json(), results[1].to_json())

    def test_goto_uses_other_fork_when_one_route_ends_at_a_parked_peer(self) -> None:
        traveller = FleetVehicleSpec(
            "traveller",
            19090,
            "traveller_spawn",
            AnchorPose(4.5, 7.5, 0.0),
        )
        solo = run_episode(
            FleetScenario("fork_solo", (traveller,), 100),
            {
                "traveller": (
                    GotoMission("cross-fork", "global_map", 16.5, 7.5, 2),
                ),
            },
            max_simulation_s=60.0,
            grid=_fork_grid(),
            linear_speed=1.0,
        )
        self.assertTrue(solo.success, solo.as_dict())

        parked = FleetVehicleSpec(
            "parked",
            19091,
            "parked_spawn",
            AnchorPose(10.5, 3.5, math.pi),
        )
        joint = run_episode(
            FleetScenario("fork_parked_peer", (traveller, parked), 100),
            {
                "traveller": (
                    GotoMission("cross-fork", "global_map", 16.5, 7.5, 2),
                ),
                "parked": (
                    GotoMission("hold-fork", "global_map", 10.5, 3.5, 2),
                ),
            },
            max_simulation_s=60.0,
            grid=_fork_grid(),
            linear_speed=1.0,
        )

        self.assert_safe_success(joint)

    def test_passing_bay_is_reachable_and_leaves_the_main_lane_clear(self) -> None:
        grid = _passing_bay_grid()
        bay_route = run_episode(
            FleetScenario(
                "passing_bay_route",
                (
                    FleetVehicleSpec(
                        "bay_vehicle",
                        19090,
                        "west_spawn",
                        AnchorPose(6.5, 5.5, 0.0),
                    ),
                ),
                100,
            ),
            {
                "bay_vehicle": (
                    GotoMission("enter-bay", "global_map", 12.5, 2.5, 2),
                    GotoMission("leave-bay", "global_map", 18.5, 5.5, 2),
                ),
            },
            max_simulation_s=60.0,
            grid=grid,
            linear_speed=1.0,
        )
        self.assertTrue(bay_route.success, bay_route.as_dict())

        holder = FleetVehicleSpec(
            "bay_holder",
            19090,
            "bay_spawn",
            AnchorPose(12.5, 2.5, 0.0),
        )
        passer = FleetVehicleSpec(
            "passer",
            19091,
            "east_spawn",
            AnchorPose(18.5, 5.5, math.pi),
        )
        passage = run_episode(
            FleetScenario("passing_bay_clearance", (holder, passer), 100),
            {
                "bay_holder": (
                    GotoMission("hold-bay", "global_map", 12.5, 2.5, 2),
                ),
                "passer": (
                    GotoMission("pass-bay", "global_map", 6.5, 5.5, 2),
                ),
            },
            max_simulation_s=60.0,
            grid=grid,
            linear_speed=1.0,
        )

        self.assert_safe_success(passage)

    def test_opposing_gotos_use_a_passing_bay_instead_of_deadlocking(self) -> None:
        specs = (
            FleetVehicleSpec(
                "westbound",
                19090,
                "east_spawn",
                AnchorPose(18.5, 5.5, math.pi),
            ),
            FleetVehicleSpec(
                "eastbound",
                19091,
                "west_spawn",
                AnchorPose(6.5, 5.5, 0.0),
            ),
        )
        missions = {
            "westbound": (
                GotoMission("go-west", "global_map", 6.5, 5.5, 2),
            ),
            "eastbound": (
                GotoMission("go-east", "global_map", 18.5, 5.5, 2),
            ),
        }
        for spec in specs:
            solo = run_episode(
                FleetScenario(f"passing_bay_{spec.vehicle_id}", (spec,), 100),
                {spec.vehicle_id: missions[spec.vehicle_id]},
                max_simulation_s=60.0,
                grid=_passing_bay_grid(),
                linear_speed=1.0,
            )
            self.assertTrue(solo.success, solo.as_dict())

        joint = run_episode(
            FleetScenario("passing_bay_swap", specs, 100),
            missions,
            max_simulation_s=90.0,
            grid=_passing_bay_grid(),
            linear_speed=1.0,
        )

        self.assert_safe_success(joint)

    def test_passing_bay_has_room_for_a_vacating_car_behind_a_deep_holder(
        self,
    ) -> None:
        specs = (
            FleetVehicleSpec(
                "vehicle_a",
                19090,
                "west_spawn",
                AnchorPose(8.5, 5.5, 0.0),
            ),
            FleetVehicleSpec(
                "vehicle_b",
                19091,
                "east_spawn",
                AnchorPose(15.5, 5.5, math.pi),
            ),
            FleetVehicleSpec(
                "vehicle_c",
                19092,
                "deep_bay_spawn",
                AnchorPose(12.0, 2.0, 0.0),
            ),
        )
        result = run_episode(
            FleetScenario("passing_bay_deep_holder", specs, 100),
            {
                "vehicle_a": (
                    GotoMission("cross-east", "global_map", 18.5, 5.5, 2),
                ),
                "vehicle_b": (
                    GotoMission("cross-west", "global_map", 6.5, 5.5, 2),
                ),
                "vehicle_c": (
                    GotoMission("hold-deep", "global_map", 12.0, 2.0, 2),
                ),
            },
            max_simulation_s=90.0,
            grid=_passing_bay_grid(),
            linear_speed=1.0,
        )

        self.assert_safe_success(result)

    def test_three_vehicle_chain_vacates_the_shallow_bay_before_main_lane(
        self,
    ) -> None:
        specs = (
            FleetVehicleSpec(
                "vehicle_a",
                19090,
                "west_spawn",
                AnchorPose(8.5, 5.5, 0.0),
            ),
            FleetVehicleSpec(
                "vehicle_b",
                19091,
                "east_spawn",
                AnchorPose(15.5, 5.5, math.pi),
            ),
            FleetVehicleSpec(
                "vehicle_c",
                19092,
                "shallow_bay_spawn",
                AnchorPose(13.5, 3.5, math.pi),
            ),
        )
        result = run_episode(
            FleetScenario("passing_bay_three_vehicle_chain", specs, 100),
            {
                "vehicle_a": (
                    GotoMission("cross-east", "global_map", 18.5, 5.5, 2),
                ),
                "vehicle_b": (
                    GotoMission("cross-west", "global_map", 6.5, 5.5, 2),
                ),
                "vehicle_c": (
                    GotoMission("join-west", "global_map", 8.5, 5.5, 2),
                ),
            },
            max_simulation_s=90.0,
            grid=_passing_bay_grid(),
            linear_speed=1.0,
        )

        self.assert_safe_success(result)


if __name__ == "__main__":
    unittest.main()
