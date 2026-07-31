"""Map-delta isolation, validation and real localhost libp2p propagation."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

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

    async def test_start_preserves_original_error_when_cleanup_also_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = P2PSettings(Path("/bin/true"), Path(temporary))
            with (
                patch.object(
                    P2PFleetSync,
                    "_start",
                    new=AsyncMock(side_effect=ValueError("original start failure")),
                ),
                patch.object(
                    P2PFleetSync,
                    "close",
                    new=AsyncMock(side_effect=RuntimeError("cleanup failure")),
                ),
                self.assertRaisesRegex(ValueError, "original start failure") as raised,
            ):
                await P2PFleetSync.start("session_1", settings, (), {})

            self.assertIsInstance(raised.exception.__cause__, RuntimeError)
            self.assertIn("cleanup failure", str(raised.exception.__cause__))

    async def test_new_runtime_and_socket_ignore_a_permissive_umask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary) / "runtime"
            runtime = self.make_runtime(runtime_dir)
            bridge = self._bridge(runtime_dir / "p1.sock")
            previous_umask = os.umask(0o002)
            try:
                runtime._acquire_runtime_lease()
                await bridge.start_server()
                config_path = runtime_dir / "vehicle_1.json"
                runtime._config_paths.append(config_path)
                with patch(
                    "mockvehicle2d.map_sync.os.fsync",
                    wraps=os.fsync,
                ) as fsync:
                    runtime._write_config(config_path, {"vehicle_id": "vehicle_1"})
                self.assertEqual(fsync.call_count, 2)
            finally:
                os.umask(previous_umask)

            self.assertEqual(stat.S_IMODE(runtime_dir.stat().st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE((runtime_dir / ".fleet.lock").stat().st_mode),
                0o600,
            )
            self.assertEqual(stat.S_IMODE(bridge.socket_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o600)
            await bridge.close()
            await runtime.close()

    async def test_existing_writable_runtime_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary) / "runtime"
            runtime_dir.mkdir(mode=0o770)
            runtime_dir.chmod(0o770)
            runtime = self.make_runtime(runtime_dir)

            with self.assertRaisesRegex(RuntimeError, "not group/world writable"):
                runtime._acquire_runtime_lease()
            await runtime.close()

    async def test_cancelled_start_reaps_identity_child_and_releases_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir(mode=0o700)
            pid_path = root / "identity.pid"
            sidecar = root / "hanging-sidecar"
            sidecar.write_text(
                "#!/usr/bin/env python3\n"
                "import os, pathlib, time\n"
                f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            sidecar.chmod(0o700)
            settings = P2PSettings(sidecar, runtime_dir, startup_timeout_s=30.0)
            vehicle = P2PVehicleConfig("vehicle_1", free_ports(1)[0], anchor(1))
            local_state = state(1)
            task = asyncio.create_task(
                P2PFleetSync.start(
                    "session_1",
                    settings,
                    (vehicle,),
                    {"vehicle_1": local_state},
                )
            )
            pid = None
            try:
                await self._wait_until(pid_path.exists)
                pid = int(pid_path.read_text(encoding="utf-8"))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

                replacement = self.make_runtime(runtime_dir)
                replacement._acquire_runtime_lease()
                await replacement.close()
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if pid is not None:
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass

    async def test_cancelled_close_finishes_cleanup_before_releasing_lease(self) -> None:
        class SlowBridge:
            def __init__(self) -> None:
                self.entered = asyncio.Event()
                self.closed = False

            async def close(self) -> None:
                self.entered.set()
                await asyncio.sleep(0.05)
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary)
            runtime = self.make_runtime(runtime_dir)
            runtime._acquire_runtime_lease()
            config_path = runtime_dir / "vehicle_1.json"
            config_path.write_text("{}", encoding="utf-8")
            runtime._config_paths.append(config_path)
            bridge = SlowBridge()
            runtime._bridges["vehicle_1"] = bridge

            close_task = asyncio.create_task(runtime.close())
            await bridge.entered.wait()
            close_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await close_task

            self.assertTrue(bridge.closed)
            self.assertFalse(config_path.exists())
            self.assertFalse(runtime._bridges)
            replacement = self.make_runtime(runtime_dir)
            replacement._acquire_runtime_lease()
            await replacement.close()

    async def test_one_bridge_close_failure_does_not_skip_other_cleanup(self) -> None:
        class Bridge:
            def __init__(self, error: Exception | None = None) -> None:
                self.error = error
                self.closed = False

            async def close(self) -> None:
                self.closed = True
                if self.error is not None:
                    raise self.error

        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary)
            runtime = self.make_runtime(runtime_dir)
            runtime._acquire_runtime_lease()
            config_path = runtime_dir / "vehicle_1.json"
            config_path.write_text("{}", encoding="utf-8")
            runtime._config_paths.append(config_path)
            failing = Bridge(RuntimeError("injected bridge failure"))
            healthy = Bridge()
            runtime._bridges.update(vehicle_1=failing, vehicle_2=healthy)

            with self.assertRaisesRegex(RuntimeError, "injected bridge failure"):
                await runtime.close()

            self.assertTrue(failing.closed)
            self.assertTrue(healthy.closed)
            self.assertFalse(config_path.exists())
            self.assertFalse(runtime._bridges)
            replacement = self.make_runtime(runtime_dir)
            replacement._acquire_runtime_lease()
            await replacement.close()

    async def test_partial_config_write_is_removed_after_start_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary) / "runtime"
            runtime_dir.mkdir(mode=0o700)
            runtime_dir.chmod(0o700)
            settings = P2PSettings(Path("/bin/true"), runtime_dir, startup_timeout_s=0.1)
            vehicle = P2PVehicleConfig("vehicle_1", free_ports(1)[0], anchor(1))

            def interrupted_write(path: Path, *_args, **_kwargs) -> int:
                with path.open("w", encoding="utf-8") as output:
                    output.write('{"protocol":')
                    output.flush()
                raise OSError("injected partial write")

            with (
                patch.object(P2PFleetSync, "_ensure_identity", new=AsyncMock(return_value="peer_1")),
                patch(
                    "mockvehicle2d.map_sync._NodeBridge.start_server",
                    new=AsyncMock(),
                ),
                patch.object(Path, "write_text", new=interrupted_write),
                patch(
                    "mockvehicle2d.map_sync.asyncio.create_subprocess_exec",
                    new=AsyncMock(side_effect=RuntimeError("injected after config")),
                ),
                self.assertRaises(Exception),
            ):
                await P2PFleetSync.start(
                    "session_1",
                    settings,
                    (vehicle,),
                    {"vehicle_1": state(1)},
                )

            self.assertFalse((runtime_dir / "vehicle_1.json").exists())
            self.assertFalse(any(".tmp" in path.name for path in runtime_dir.iterdir()))

    async def test_atomic_config_failure_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary)
            settings = P2PSettings(Path("/bin/true"), runtime_dir, startup_timeout_s=0.1)
            vehicle = P2PVehicleConfig("vehicle_1", free_ports(1)[0], anchor(1))

            with (
                patch.object(P2PFleetSync, "_ensure_identity", new=AsyncMock(return_value="peer_1")),
                patch(
                    "mockvehicle2d.map_sync._NodeBridge.start_server",
                    new=AsyncMock(),
                ),
                patch("mockvehicle2d.map_sync.os.replace", side_effect=OSError("injected replace")) as replace,
                self.assertRaisesRegex(OSError, "injected replace"),
            ):
                await P2PFleetSync.start(
                    "session_1",
                    settings,
                    (vehicle,),
                    {"vehicle_1": state(1)},
                )

            replace.assert_called_once()
            self.assertFalse((runtime_dir / "vehicle_1.json").exists())
            self.assertFalse(any(".tmp" in path.name for path in runtime_dir.iterdir()))

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

    async def test_cancelled_bridge_close_still_removes_socket_and_reaps_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "p1.sock"
            bridge = self._bridge(path)
            await bridge.start_server()
            _reader, writer = await asyncio.open_unix_connection(path)
            await self._wait_until(lambda: bridge._writer is not None)
            bridge.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import time; time.sleep(0.1)",
            )
            close_task = asyncio.create_task(bridge.close())
            await asyncio.sleep(0.01)
            close_task.cancel()
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await close_task
                self.assertIsNotNone(bridge.process.returncode)
                self.assertFalse(path.exists())
            finally:
                if bridge.process.returncode is None:
                    bridge.process.kill()
                    await bridge.process.wait()
                await bridge.close()
                writer.close()
                await writer.wait_closed()

    def _bridge(self, path: Path, vehicle_id: str = "vehicle_1"):
        from mockvehicle2d.map_sync import _NodeBridge

        return _NodeBridge(vehicle_id, path, state(1))

    async def _wait_until(self, predicate, timeout_s: float = 2.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                self.fail("timed out waiting for test condition")
            await asyncio.sleep(0.01)


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
            private_artifacts = [Path(temporary.name) / ".fleet.lock"]
            for index, spec in enumerate(scenario.vehicles, 1):
                private_artifacts.extend(
                    (
                        Path(temporary.name) / f"{spec.vehicle_id}.json",
                        Path(temporary.name) / f"{spec.vehicle_id}.key",
                        Path(temporary.name) / f"p{index}.sock",
                    )
                )
            self.assertTrue(all(path.exists() for path in private_artifacts))
            self.assertTrue(
                all(
                    stat.S_IMODE(path.stat().st_mode) == 0o600
                    for path in private_artifacts
                )
            )

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
