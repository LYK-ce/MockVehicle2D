"""Map-delta isolation, validation and real localhost libp2p propagation."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock

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
        self.assertTrue(local.snapshot()["collaborative_view_current"])
        self.assertEqual(local.snapshot()["collaborative_known_cells"], 2)

    def test_snapshot_does_not_materialize_the_collaborative_view(self) -> None:
        local = state(1)
        local.configure_network("peer_1", {"vehicle_2": ("peer_2", anchor(2))})
        local.record_local(LocalMapDelta((MapCellUpdate(0, 0, OCCUPIED),)))
        local.collaborative_cells = Mock(side_effect=AssertionError("hot-path projection"))
        local._project = Mock(side_effect=AssertionError("hot-path transform"))

        snapshot = local.snapshot()

        self.assertEqual(snapshot["own_known_cells"], 1)
        self.assertEqual(snapshot["collaborative_evidence_cells"], 1)
        self.assertFalse(snapshot["collaborative_view_current"])
        self.assertIsNone(snapshot["collaborative_known_cells"])
        local.collaborative_cells.assert_not_called()
        local._project.assert_not_called()

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


class TestP2PRuntimeOwnership(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, runtime_dir: Path) -> P2PFleetSync:
        settings = P2PSettings(Path("/bin/true"), runtime_dir)
        return P2PFleetSync("session_1", settings, (), {})

    async def test_runtime_directory_has_one_live_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = self.make_runtime(Path(temporary))
            second = self.make_runtime(Path(temporary))
            first._acquire_runtime_lease()

            with self.assertRaisesRegex(RuntimeError, "already in use"):
                second._acquire_runtime_lease()
            await second.close()

            third = self.make_runtime(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                third._acquire_runtime_lease()
            await first.close()
            third._acquire_runtime_lease()
            await third.close()

    async def test_bridge_refuses_to_remove_a_non_socket_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file_path = root / "p1.sock"
            file_path.write_text("keep me", encoding="utf-8")
            directory_path = root / "p2.sock"
            directory_path.mkdir()

            for path in (file_path, directory_path):
                bridge = self._bridge(path)
                with self.subTest(path=path), self.assertRaisesRegex(
                    RuntimeError,
                    "not a Unix socket",
                ):
                    await bridge.start_server()
                await bridge.close()

            self.assertEqual(file_path.read_text(encoding="utf-8"), "keep me")
            self.assertTrue(directory_path.is_dir())

    async def test_bridge_replaces_stale_socket_and_removes_only_its_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "p1.sock"
            stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stale.bind(str(path))
            stale.close()
            bridge = self._bridge(path)

            await bridge.start_server()
            self.assertTrue(path.is_socket())
            await bridge.close()

            self.assertFalse(path.exists())

    async def test_failed_bridge_close_does_not_remove_live_owner_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "p1.sock"
            owner = self._bridge(path, vehicle_id="vehicle_1")
            contender = self._bridge(path, vehicle_id="vehicle_2")
            await owner.start_server()

            with self.assertRaisesRegex(RuntimeError, "already active"):
                await contender.start_server()
            await contender.close()
            self.assertTrue(path.is_socket())

            _reader, writer = await asyncio.open_unix_connection(path)
            writer.close()
            await writer.wait_closed()
            await owner.close()

    def _bridge(self, path: Path, vehicle_id: str = "vehicle_1"):
        from mockvehicle2d.map_sync import _NodeBridge

        return _NodeBridge(vehicle_id, path, state(1))


@unittest.skipUnless(SIDECAR.is_file(), "build map-sync-node before live libp2p test")
class TestLiveLibp2pMesh(unittest.IsolatedAsyncioTestCase):
    async def test_four_peers_propagate_and_survive_one_peer_exit(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        from mockvehicle2d.fleet import (
            AnchorPose,
            FleetRuntime,
            FleetScenario,
            FleetVehicleSpec,
            fleet_handler,
        )

        temporary = tempfile.TemporaryDirectory()
        runtime: P2PFleetSync | None = None
        servers = []
        connections = []
        try:
            ports = free_ports(8)
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
                        ports[number - 1],
                        f"spawn_{number}",
                        AnchorPose(number * 10.0, number * 5.0, 0.0),
                        ports[number + 3],
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

            for spec in scenario.vehicles:
                async def configured_handler(
                    websocket,
                    vehicle_id: str = spec.vehicle_id,
                ) -> None:
                    await fleet_handler(websocket, fleet=fleet, vehicle_id=vehicle_id)

                servers.append(await serve(configured_handler, "127.0.0.1", spec.operator_port))
            for spec in scenario.vehicles:
                connections.append(await connect(f"ws://127.0.0.1:{spec.operator_port}"))
            hellos = [json.loads(await connection.recv()) for connection in connections]
            self.assertEqual(
                {hello["vehicle_id"] for hello in hellos},
                {spec.vehicle_id for spec in scenario.vehicles},
            )
            self.assertTrue(all(hello["protocol_version"] == 4 for hello in hellos))

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
            await asyncio.gather(
                *(connection.close() for connection in connections),
                return_exceptions=True,
            )
            for server in servers:
                server.close()
            if servers:
                await asyncio.gather(*(server.wait_closed() for server in servers))
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
