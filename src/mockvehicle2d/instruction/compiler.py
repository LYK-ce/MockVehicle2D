"""Task compiler — converts validated instructions into executable task descriptions.

Phase 1: compiles instructions into task dicts without controlling the vehicle.
Phase 2: will actually invoke vehicle / navigation / safety methods.
The optional truth-grid A* branch is retained only for offline simulator debugging.
"""

from __future__ import annotations

from typing import Any

from mockvehicle2d.collision import is_swept_circle_passable
from mockvehicle2d.map_grid import MapGrid


class TaskCompiler:
    """Convert validated instructions into executable task dicts.

    Each compile_* method returns a dict describing the action to take.
    In Phase 1 these do NOT modify vehicle state — they describe what WOULD happen.
    When explicitly constructed with a simulator truth grid, this legacy/debug
    compiler can produce an A* reference path. The production WebSocket handler
    never executes that controller choice; it routes navigation to finite-view
    D* Lite.
    """

    def __init__(self, grid: MapGrid | None = None) -> None:
        self.grid = grid

    # ── public API ───────────────────────────────────────────

    def compile(self, instruction: dict, state_snapshot: dict | None = None) -> dict:
        """Compile a validated instruction into a task dict.

        Parameters
        ----------
        instruction : dict
            A validated instruction with intent, parameters, timestamp, etc.
        state_snapshot : dict | None
            Optional current state (pose, navigation, safety) for context.

        Returns
        -------
        dict
            Task description with type, action, and metadata.
        """
        intent = instruction.get("intent", "clarify")
        params = instruction.get("parameters", {}) or {}
        method = getattr(self, f"_compile_{intent}", None)
        if method is None:
            return self._make_task("unknown", {"intent": intent})
        return method(params, state_snapshot or {})

    def with_grid(self, grid: MapGrid) -> "TaskCompiler":
        """Return a new compiler sharing the same state but with a different grid.

        Useful for server.py which discovers the grid after compiler construction.
        """
        new = TaskCompiler(grid)
        return new

    # ── per-intent compilers ─────────────────────────────────

    def _compile_stop(self, params: dict, snapshot: dict) -> dict:
        return self._make_task("immediate", {
            "action": "stop",
            "cancel_active_task": True,
        })

    def _compile_goto(self, params: dict, snapshot: dict) -> dict:
        x_m, y_m = params["x_m"], params["y_m"]
        # Legacy simulator-debug truth-grid branch.
        pose = snapshot.get("pose", {})
        start_x = pose.get("x_m", 0.0)
        start_y = pose.get("y_m", 0.0)
        if self.grid is not None:
            if not _is_straight_path_clear(self.grid, start_x, start_y, x_m, y_m):
                path = self._plan_path(start_x, start_y, x_m, y_m)
                if path is not None:
                    return self._make_task("navigation", {
                        "action": "goto",
                        "controller": "PathFollowingController",
                        "goal": {"x_m": x_m, "y_m": y_m},
                        "path": path,
                    })
                return self._make_task("navigation", {
                    "action": "goto",
                    "controller": "blocked",
                    "goal": {"x_m": x_m, "y_m": y_m},
                    "reason": "no path found",
                })
        return self._make_task("navigation", {
            "action": "goto",
            "controller": "GotoController",
            "goal": {"x_m": x_m, "y_m": y_m},
        })

    def _compile_patrol(self, params: dict, snapshot: dict) -> dict:
        return self._make_task("navigation", {
            "action": "patrol",
        })

    def _compile_clarify(self, params: dict, snapshot: dict) -> dict:
        question = params.get("question", "")
        missing = params.get("missing_parameters", [])
        return self._make_task("clarification", {
            "action": "clarify",
            "question": question,
            "missing_parameters": missing,
        })

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _make_task(task_type: str, details: dict[str, Any]) -> dict:
        return {"type": task_type, **details}

    # ── Legacy simulator-debug path planning helpers ─────────

    def _plan_path(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
    ) -> list[dict[str, float]] | None:
        """Run A* internally and return SI metre waypoints.

        Returns labelled metric coordinates or ``None`` when no path exists.
        """
        if self.grid is None:
            return None
        from mockvehicle2d.pathfinding.a_star import a_star_search

        sx = int(round(start_x))
        sy = int(round(start_y))
        gx = int(round(goal_x))
        gy = int(round(goal_y))
        path = a_star_search(self.grid, (sx, sy), (gx, gy))
        if path is None:
            return None
        return [
            {"x_m": float(cell_x), "y_m": float(cell_y)}
            for cell_x, cell_y in path
        ]


# ── module-level helper ──────────────────────────────────────


def _is_straight_path_clear(
    grid: MapGrid,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float = 0.5,
) -> bool:
    """Check if the straight line from (x1,y1) to (x2,y2) is obstacle-free."""
    return is_swept_circle_passable(grid, x1, y1, x2, y2, radius)
