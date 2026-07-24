"""Task compiler — converts validated instructions into executable task descriptions.

Phase 1: compiles instructions into task dicts without controlling the vehicle.
Phase 2: will actually invoke vehicle / navigation / safety methods.
"""

from __future__ import annotations

import math
from typing import Any


class TaskCompiler:
    """Convert validated instructions into executable task dicts.

    Each compile_* method returns a dict describing the action to take.
    In Phase 1 these do NOT modify vehicle state — they describe what WOULD happen.
    """

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

    # ── per-intent compilers ─────────────────────────────────

    def _compile_stop(self, params: dict, snapshot: dict) -> dict:
        return self._make_task("immediate", {
            "action": "stop",
            "cancel_active_task": True,
        })

    def _compile_status(self, params: dict, snapshot: dict) -> dict:
        return self._make_task("query", {
            "action": "status",
            "snapshot": snapshot,
        })

    def _compile_goto_point(self, params: dict, snapshot: dict) -> dict:
        x_m, y_m = params["x_m"], params["y_m"]
        return self._make_task("navigation", {
            "action": "goto_point",
            "controller": "GotoController",
            "goal": {"x_m": x_m, "y_m": y_m},
        })

    def _compile_move_distance(self, params: dict, snapshot: dict) -> dict:
        distance_m = params["distance_m"]
        direction = params["direction"]
        # Compute relative goal from current pose if available
        pose = snapshot.get("pose", {})
        current_x = pose.get("x", 0.0)
        current_y = pose.get("y", 0.0)
        current_yaw = pose.get("yaw", 0.0)
        sign = 1.0 if direction == "forward" else -1.0
        goal_x = current_x + sign * distance_m * math.cos(current_yaw)
        goal_y = current_y + sign * distance_m * math.sin(current_yaw)
        return self._make_task("navigation", {
            "action": "move_distance",
            "controller": "GotoController",
            "distance_m": distance_m,
            "direction": direction,
            "goal": {"x_m": round(goal_x, 4), "y_m": round(goal_y, 4)},
        })

    def _compile_rotate(self, params: dict, snapshot: dict) -> dict:
        angle_deg = params["angle_deg"]
        direction = params["direction"]
        # Compute target yaw
        pose = snapshot.get("pose", {})
        current_yaw = pose.get("yaw", 0.0)
        sign = 1.0 if direction == "left" else -1.0
        target_yaw = current_yaw + sign * math.radians(angle_deg)
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
        return self._make_task("rotation", {
            "action": "rotate",
            "angle_deg": angle_deg,
            "direction": direction,
            "target_yaw_rad": round(target_yaw, 4),
        })

    def _compile_scan_report(self, params: dict, snapshot: dict) -> dict:
        query = params.get("query", "")
        scan_data = snapshot.get("scan", {})
        summary = self._summarize_scan(scan_data, query)
        return self._make_task("query", {
            "action": "scan_report",
            "query": query,
            "summary": summary,
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

    @staticmethod
    def _summarize_scan(scan_data: dict, query: str = "") -> dict[str, Any]:
        """Summarize a scan frame: obstacle distances by sector.

        In Phase 1 this uses scan_data dict; in Phase 2 it will consume
        real LaserPoint lists from the scanner.
        """
        points = scan_data.get("points", [])
        if not points:
            return {"sectors": {}, "total_points": 0}

        sectors = {"front": [], "left": [], "right": [], "back": []}
        for pt in points:
            angle = pt.get("angle", 0.0)
            rng = pt.get("range", 0.0)
            if rng <= 0:
                continue
            # sector classification by angle (forward = 0 rad, +x)
            if -math.pi / 4 <= angle < math.pi / 4:
                sectors["front"].append(rng)
            elif math.pi / 4 <= angle < 3 * math.pi / 4:
                sectors["left"].append(rng)
            elif -3 * math.pi / 4 <= angle < -math.pi / 4:
                sectors["right"].append(rng)
            else:
                sectors["back"].append(rng)

        summary = {}
        for sector, ranges in sectors.items():
            if ranges:
                summary[sector] = {
                    "min_m": round(min(ranges), 2),
                    "avg_m": round(sum(ranges) / len(ranges), 2),
                    "count": len(ranges),
                }
            else:
                summary[sector] = {"min_m": None, "avg_m": None, "count": 0}

        return {"sectors": summary, "total_points": len(points)}
