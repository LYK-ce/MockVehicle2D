"""Command-line pacing defaults and validation."""

import argparse
import importlib
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

cli = importlib.import_module("mockvehicle2d.cli.main")


class TestRealtimeFactorCli(unittest.TestCase):
    def test_serve_and_fleet_default_to_five_times_realtime(self) -> None:
        from mockvehicle2d import fleet, server

        cases = (
            ("serve", server, ("serve",)),
            ("fleet", fleet, ("fleet", "--scenario", "scenario.json")),
        )
        for name, module, arguments in cases:
            with (
                self.subTest(command=name),
                patch.object(sys, "argv", ["mockvehicle2d", *arguments]),
                patch.object(module, "main", new=Mock(return_value=None)) as target,
                patch.object(cli.asyncio, "run") as run,
            ):
                cli.main()

            self.assertEqual(target.call_args.kwargs["realtime_factor"], 5.0)
            run.assert_called_once_with(None)

    def test_realtime_factor_accepts_one_and_rejects_nonpositive_or_nonfinite(
        self,
    ) -> None:
        self.assertEqual(cli._positive_float("1"), 1.0)
        for value in ("0", "-1", str(math.inf), str(math.nan)):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                cli._positive_float(value)


if __name__ == "__main__":
    unittest.main()
