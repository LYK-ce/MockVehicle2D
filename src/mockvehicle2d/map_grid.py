"""
mock_map_grid.py — 2D 占据栅格地图

MapGrid 使用 1D bytearray 存储，支持:
  - 点查询 O(1)
  - 从 voxel 数组批量构造
  - 包围盒遍历（用于圆形碰撞检测）
"""

from typing import Optional


class MapGrid:
    """2D 占据栅格地图

    cells[y * width + x]:
      0 = 可通行 (free)
      1 = 不可通行 (wall)
    """

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid grid size: {width}x{height}")
        self.width = width
        self.height = height
        self._cells = bytearray(width * height)

    # ── 点操作 ─────────────────────────────────────────

    def set_cell(self, x: int, y: int, state: int) -> None:
        if not self.in_bounds(x, y):
            raise IndexError(f"Cell ({x},{y}) out of bounds ({self.width}x{self.height})")
        self._cells[y * self.width + x] = state

    def get_cell(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            raise IndexError(f"Cell ({x},{y}) out of bounds ({self.width}x{self.height})")
        return self._cells[y * self.width + x]

    def is_wall(self, x: int, y: int) -> bool:
        return self.get_cell(x, y) == 1

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x: int, y: int) -> bool:
        """点是否可通行（在界内且非墙）"""
        return self.in_bounds(x, y) and not self.is_wall(x, y)

    # ── 批量构造 ───────────────────────────────────────

    @classmethod
    def from_voxels(cls, voxels: list[dict]) -> "MapGrid":
        """从 voxel 数组构造 MapGrid，自动推导尺寸。

        每个 voxel: {"gx": int, "gy": int, "state": int, ...}
        """
        if not voxels:
            return cls(1, 1)

        max_x = max(v["gx"] for v in voxels)
        max_y = max(v["gy"] for v in voxels)
        grid = cls(max_x + 1, max_y + 1)

        for v in voxels:
            grid.set_cell(v["gx"], v["gy"], v.get("state", 0))

        return grid

    @classmethod
    def from_wall_set(cls, width: int, height: int, walls: set[tuple[int, int]]) -> "MapGrid":
        """从墙坐标集合构造（测试用）"""
        grid = cls(width, height)
        for x, y in walls:
            grid.set_cell(x, y, 1)
        return grid

    # ── 统计 ───────────────────────────────────────────

    def __repr__(self) -> str:
        wall_count = sum(1 for c in self._cells if c == 1)
        return f"MapGrid({self.width}x{self.height}, walls={wall_count})"
