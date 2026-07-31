"""Map-delta isolation, validation and real localhost libp2p propagation."""

import asyncio
from copy import deepcopy
from pathlib import Path
import socket
import tempfile
import unittest

from mockvehicle2d.local_state import (
    FREE,
    OCCUPIED,
    AnchorSpec,
    LocalMapDelta,
    MapCellUpdate,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.map_sync import (
    MAX_DELTA_CELLS,
    MapSyncState,
    P2PFleetSync,
    P2PSettings,
    P2PVehicleConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIDECAR = REPO_ROOT / "target" / "debug" / "map-sync-node"


def anchor(number: int) -> AnchorSpec:
    return AnchorSpec(f"spawn_{number}", number * 10.0, number * 5.0, 0.0)


def state(number: int) -> MapSyncState:
    return MapSyncState("session_1", f"vehicle_{number}", anchor(number), 1.0)


class TestMapSyncState(unittest.TestCase):
    def test_dirty_cells_are_merged_bounded_and_acknowledged(self) -> None:
        local = state(1)
        local.configure_network("peer_1", {"vehicle_2": ("peer_2", anchor(2))})
        local.record_local(
            LocalMapDelta(
                tuple(
                    MapCellUpdate(index, 0, FREE)
                    for index in range(MAX_DELTA_CELLS + 2)
                )
            )
        )
        local.record_local(LocalMapDelta((MapCellUpdate(0, 0, OCCUPIED),)))

        first = local.prepare_delta()

        self.assertIsNotNone(first)
        self.assertEqual(len(first["cells"]), MAX_DELTA_CELLS)
        self.assertEqual(first["cells"][0], {"gx": 0, "gy": 0, "state": OCCUPIED})
        self.assertIsNone(local.prepare_delta())
        local.publish_result(first["sequence"], True)
        self.assertEqual(local.dirty_count, 2)
        second = local.prepare_delta()
        self.assertEqual(second["sequence"], 2)

    def test_remote_evidence_is_validated_deduplicated_and_never_republished(self) -> None:
        source = state(1)
        receiver = state(2)
        source.configure_network("peer_1", {"vehicle_2": ("peer_2", anchor(2))})
        receiver.configure_network("peer_2", {"vehicle_1": ("peer_1", anchor(1))})
        source.record_local(LocalMapDelta((MapCellUpdate(3, 4, OCCUPIED),)))
        payload = source.prepare_delta()

        self.assertTrue(receiver.receive("peer_1", "vehicle_1", payload))
        self.assertEqual(receiver.peer_evidence("vehicle_1"), {(3, 4): OCCUPIED})
        self.assertFalse(receiver.receive("peer_1", "vehicle_1", payload))
        self.assertIsNone(receiver.prepare_delta())
        self.assertEqual(receiver.received_deltas, 1)
        self.assertEqual(receiver.rejected_deltas, 1)

        wrong_session = deepcopy(payload)
        wrong_session["sequence"] = 2
        wrong_session["session_id"] = "other"
        self.assertFalse(receiver.receive("peer_1", "vehicle_1", wrong_session))
        self.assertFalse(receiver.receive("unknown", "vehicle_1", payload))
        self.assertEqual(receiver.peer_evidence("vehicle_1"), {(3, 4): OCCUPIED})

    def test_collaborative_view_projects_sources_without_changing_own_map(self) -> None:
        local = state(2)
        local.configure_network("peer_2", {"vehicle_1": ("peer_1", anchor(1))})
        local.record_local(LocalMapDelta((MapCellUpdate(0, 0, FREE),)))
        source = state(1)
        source.configure_network("peer_1", {"vehicle_2": ("peer_2", anchor(2))})
        source.record_local(LocalMapDelta((MapCellUpdate(0, 0, OCCUPIED),)))

        self.assertTrue(local.receive("peer_1", "vehicle_1", source.prepare_delta()))

        combined = local.collaborative_cells()
        self.assertEqual(combined[(10, 5)], OCCUPIED)
        self.assertEqual(combined[(20, 10)], FREE)
        self.assertEqual(local.dirty_count, 1)

    def test_settings_reject_unknown_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            P2PSettings.from_json({"sidecar_path": "node"})
        with self.assertRaises(ValueError):
            P2PSettings.from_json(
                {"sidecar_path": "node", "runtime_dir": "run", "extra": True}
            )
        with self.assertRaises(ValueError):
            P2PSettings(Path("node"), Path("run"), sync_interval_ms=0)


def free_ports(count: int) -> list[int]:
    sockets = []
    try:
        for _ in range(count):
            candidate = socket.socket()
            candidate.bind(("127.0.0.1", 0))
            sockets.append(candidate)
        return [candidate.getsockname()[1] for candidate in sockets]
    finally:
        for candidate in sockets:
            candidate.close()


@unittest.skipUnless(SIDECAR.is_file(), "build map-sync-node before live libp2p test")
class TestLiveLibp2pMesh(unittest.IsolatedAsyncioTestCase):
    async def test_four_peers_propagate_and_survive_one_peer_exit(self) -> None:
        from mockvehicle2d.fleet import AnchorPose, FleetRuntime, FleetScenario, FleetVehicleSpec

        temporary = tempfile.TemporaryDirectory()
        runtime: P2PFleetSync | None = None
        try:
            ports = free_ports(4)
            settings = P2PSettings(
                SIDECAR,
                Path(temporary.name),
                startup_timeout_s=15.0,
            )
            scenario = FleetScenario(
                "session_1",
                tuple(
                    FleetVehicleSpec(
                        f"vehicle_{number}",
                        19089 + number,
                        f"spawn_{number}",
                        AnchorPose(number * 10.0, number * 5.0, 0.0),
                        ports[number - 1],
                    )
                    for number in range(1, 5)
                ),
                100,
                settings,
            )
            fleet = FleetRuntime.create(scenario, grid=MapGrid(60, 60))
            vehicles = tuple(
                P2PVehicleConfig(
                    spec.vehicle_id,
                    spec.p2p_port,
                    fleet.nodes[spec.vehicle_id].local_state.anchor,
                )
                for spec in scenario.vehicles
            )
            states = {
                vehicle_id: node.map_sync
                for vehicle_id, node in fleet.nodes.items()
            }
            runtime = await P2PFleetSync.start(
                "session_1",
                settings,
                vehicles,
                states,
            )
            self.assertEqual(len({item.local_peer_id for item in states.values()}), 4)

            states["vehicle_1"].record_local(
                LocalMapDelta((MapCellUpdate(7, 8, OCCUPIED),))
            )
            await self._wait_until(
                lambda: all(
                    states[vehicle_id].peer_evidence("vehicle_1").get((7, 8)) == OCCUPIED
                    for vehicle_id in ("vehicle_2", "vehicle_3", "vehicle_4")
                )
            )

            failed = runtime._bridges["vehicle_4"].process
            failed.terminate()
            await failed.wait()
            await self._wait_until(lambda: not states["vehicle_4"].ready)
            fleet.tick(0.1)
            states["vehicle_1"].record_local(
                LocalMapDelta((MapCellUpdate(9, 8, OCCUPIED),))
            )
            await self._wait_until(
                lambda: all(
                    states[vehicle_id].peer_evidence("vehicle_1").get((9, 8)) == OCCUPIED
                    for vehicle_id in ("vehicle_2", "vehicle_3")
                )
            )
        finally:
            if runtime is not None:
                await runtime.close()
            temporary.cleanup()

    async def _wait_until(self, predicate, timeout_s: float = 8.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                self.fail("timed out waiting for libp2p propagation")
            await asyncio.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
