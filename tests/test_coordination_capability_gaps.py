"""Public Episode scenarios that expose coordination capability limits."""

import math
import unittest

from mockvehicle2d.controller import GotoMission
from mockvehicle2d.episode import EpisodeResult, run_episode
from mockvehicle2d.fleet import AnchorPose, FleetScenario, FleetVehicleSpec
from mockvehicle2d.map_grid import MapGrid


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


class TestCoordinationCapabilityGaps(unittest.TestCase):
    def assert_safe_success(self, result: EpisodeResult) -> None:
        self.assertTrue(
            result.success
            and all(not vehicle["collision_occurred"] for vehicle in result.vehicles),
            result.as_dict(),
        )

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
