#!/usr/bin/env python3
"""
test_collision.py — 碰撞检测测试用例

车辆参数: 圆形, radius=0.5 (直径=1 cell), 有航向角 yaw
地图: 1 cell = 1m × 1m

覆盖:
  - MapGrid: 基础操作、构造方法、边界情况
  - raycast: Bresenham 线段碰撞 (9 个场景)
  - is_circle_passable: 圆形碰撞 r=0.5 (9 个场景)
  - 航向角: pose 更新 + 碰撞联动 (4 个场景)

用法:
  cd test_tool && python test_collision.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mock_map_grid import MapGrid
from mock_collision import raycast, is_circle_passable, CollisionResult


PASS = 0
FAIL = 0
R = 0.5  # 车辆半径


def test(name: str):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


def check(condition: bool, msg: str):
    global PASS, FAIL
    if condition:
        print(f"  ✅ {msg}")
        PASS += 1
    else:
        print(f"  ❌ {msg}")
        FAIL += 1


def make_grid(width: int, height: int, walls: set) -> MapGrid:
    return MapGrid.from_wall_set(width, height, walls)


# ═══════════════════════════════════════════════════════
#  MapGrid 测试
# ═══════════════════════════════════════════════════════

def test_mapgrid_basic():
    test("MapGrid — 基础 get/set")
    g = MapGrid(10, 10)
    check(g.get_cell(0, 0) == 0, "新 grid 全部为 0")
    g.set_cell(3, 4, 1)
    check(g.get_cell(3, 4) == 1, "set_cell(3,4,1) → get 返回 1")
    check(g.get_cell(3, 5) == 0, "相邻格子不受影响")
    g.set_cell(3, 4, 0)
    check(g.get_cell(3, 4) == 0, "可覆盖为 0")


def test_mapgrid_wall_check():
    test("MapGrid — is_wall / is_passable")
    g = MapGrid(10, 10)
    g.set_cell(5, 5, 1)
    check(g.is_wall(5, 5) == True, "is_wall(5,5) → True")
    check(g.is_wall(5, 6) == False, "is_wall(5,6) → False")
    check(g.is_passable(5, 5) == False, "is_passable(5,5) → False（墙）")
    check(g.is_passable(5, 6) == True, "is_passable(5,6) → True")
    check(g.is_passable(-1, 0) == False, "is_passable(-1,0) → False（越界）")
    check(g.is_passable(10, 0) == False, "is_passable(10,0) → False（越界）")


def test_mapgrid_bounds():
    test("MapGrid — in_bounds")
    g = MapGrid(10, 10)
    check(g.in_bounds(0, 0) == True, "(0,0) 在界内")
    check(g.in_bounds(9, 9) == True, "(9,9) 在界内")
    check(g.in_bounds(-1, 0) == False, "(-1,0) 越界")
    check(g.in_bounds(0, 10) == False, "(0,10) 越界")


def test_mapgrid_from_voxels():
    test("MapGrid — from_voxels 构造")
    voxels = [
        {"gx": 0, "gy": 0, "state": 1},
        {"gx": 1, "gy": 2, "state": 1},
        {"gx": 4, "gy": 4, "state": 0},
    ]
    g = MapGrid.from_voxels(voxels)
    check(g.width == 5, f"宽度 → 5 (实际 {g.width})")
    check(g.height == 5, f"高度 → 5 (实际 {g.height})")
    check(g.is_wall(0, 0) == True, "(0,0) 为墙")
    check(g.is_wall(1, 2) == True, "(1,2) 为墙")
    check(g.is_wall(4, 4) == False, "(4,4) state=0 → 非墙")


def test_mapgrid_from_wall_set():
    test("MapGrid — from_wall_set 构造")
    g = make_grid(10, 10, {(0, 0), (9, 9), (5, 5)})
    check(g.is_wall(0, 0) == True, "(0,0) 为墙")
    check(g.is_wall(9, 9) == True, "(9,9) 为墙")
    check(g.is_wall(5, 5) == True, "(5,5) 为墙")
    check(g.is_wall(1, 1) == False, "(1,1) 非墙")


def test_mapgrid_empty_voxels():
    test("MapGrid — 空 voxel 数组")
    g = MapGrid.from_voxels([])
    check(g.width == 1, "空数组 → 1x1 grid")


def test_mapgrid_oob_error():
    test("MapGrid — 越界报错")
    g = MapGrid(5, 5)
    try:
        g.set_cell(10, 10, 1)
        check(False, "应抛出 IndexError")
    except IndexError:
        check(True, "越界 set_cell 抛出 IndexError")
    try:
        g.get_cell(10, 10)
        check(False, "应抛出 IndexError")
    except IndexError:
        check(True, "越界 get_cell 抛出 IndexError")


def test_mapgrid_large():
    test("MapGrid — 大规模性能")
    import time
    t0 = time.time()
    g = MapGrid(1000, 1000)
    for i in range(10000):
        g.set_cell(i % 1000, i // 1000, 1)
    elapsed = time.time() - t0
    check(elapsed < 1.0, f"10000 set_cell 耗时 {elapsed*1000:.1f}ms < 1s")


# ═══════════════════════════════════════════════════════
#  Raycast (Bresenham) 测试
# ═══════════════════════════════════════════════════════

def test_raycast_horizontal_clear():
    test("Raycast — 水平直线，无碰撞")
    g = make_grid(10, 10, set())
    r = raycast(g, 0, 5, 9, 5)
    check(r.hit == False, "(0,5)→(9,5) 通过")


def test_raycast_vertical_clear():
    test("Raycast — 垂直直线，无碰撞")
    g = make_grid(10, 10, set())
    r = raycast(g, 3, 0, 3, 9)
    check(r.hit == False, "(3,0)→(3,9) 通过")


def test_raycast_diagonal_clear():
    test("Raycast — 对角线，无碰撞")
    g = make_grid(10, 10, set())
    r = raycast(g, 0, 0, 9, 9)
    check(r.hit == False, "(0,0)→(9,9) 通过")


def test_raycast_wall_at_endpoint():
    test("Raycast — 终点是墙")
    g = make_grid(10, 10, {(5, 5)})
    r = raycast(g, 0, 0, 5, 5)
    check(r.hit == True, "检测到碰撞")
    check(r.x == 5 and r.y == 5, f"碰撞点 (5,5) 实际 ({r.x},{r.y})")


def test_raycast_wall_in_middle():
    test("Raycast — 路径中间有墙")
    g = make_grid(10, 10, {(5, 3)})
    r = raycast(g, 0, 3, 9, 3)
    check(r.hit == True, "检测到碰撞")
    check(r.x == 5 and r.y == 3, f"碰撞点 (5,3) 实际 ({r.x},{r.y})")


def test_raycast_wall_immediate():
    test("Raycast — 起点就是墙")
    g = make_grid(10, 10, {(0, 0)})
    r = raycast(g, 0, 0, 9, 9)
    check(r.hit == True, "检测到碰撞")
    check(r.x == 0 and r.y == 0, f"碰撞点 (0,0) 实际 ({r.x},{r.y})")


def test_raycast_single_cell():
    test("Raycast — 起点=终点")
    g = make_grid(10, 10, set())
    r = raycast(g, 3, 3, 3, 3)
    check(r.hit == False, "空地, 通过")
    g2 = make_grid(10, 10, {(3, 3)})
    r2 = raycast(g2, 3, 3, 3, 3)
    check(r2.hit == True, "墙地, 碰撞")


def test_raycast_negative_direction():
    test("Raycast — 负方向移动")
    g = make_grid(10, 10, {(3, 3)})
    r = raycast(g, 9, 9, 0, 0)
    check(r.hit == True, "(9,9)→(0,0) 途经 (3,3) 碰撞")


def test_raycast_grazing():
    test("Raycast — 紧贴墙经过")
    g = make_grid(10, 5, {(5, 0)})
    r = raycast(g, 0, 1, 9, 1)
    check(r.hit == False, "从墙旁边经过，通过")


# ═══════════════════════════════════════════════════════
#  圆形碰撞测试 — 车辆 radius=0.5
# ═══════════════════════════════════════════════════════

def test_circle_open_center():
    test("圆形 r=0.5 — 空地中央")
    g = make_grid(20, 20, set())
    check(is_circle_passable(g, 10.0, 10.0, R) == True,
          "(10,10) 空旷 → 通过")


def test_circle_on_wall():
    test("圆形 r=0.5 — 圆心所在格子是墙")
    g = make_grid(20, 20, {(10, 10)})
    check(is_circle_passable(g, 10.0, 10.0, R) == False,
          "圆心在墙上 → 碰撞")
    check(is_circle_passable(g, 10.3, 10.3, R) == False,
          "圆心偏移仍在 cell(10,10) → 碰撞")


def test_circle_beside_wall_safe():
    test("圆形 r=0.5 — 贴着墙但未侵入")
    g = make_grid(20, 20, {(6, 5)})
    check(is_circle_passable(g, 5.0, 5.0, R) == True,
          "墙在隔壁 cell, 距离 1 > 0.5 → 通过")


def test_circle_near_boundary_collision():
    test("圆形 r=0.5 — 越界检测")
    g = make_grid(10, 10, set())
    # 圆心 x=0.3 → 圆覆盖 [-0.2, 0.8]
    # cell(-1,5) [−1,0]: closest=(0,5), dx=-0.3, d²=0.09 < 0.25 → 重叠且越界!
    check(is_circle_passable(g, 0.3, 5.0, R) == False,
          "圆心 x=0.3 → cell(-1,5) 重叠且越界 → 碰撞")


def test_circle_near_boundary_safe():
    test("圆形 r=0.5 — 靠近边界但安全")
    g = make_grid(10, 10, set())
    check(is_circle_passable(g, 1.0, 5.0, R) == True,
          "圆心 x=1.0, 圆 [0.5,1.5] 全在界内 → 通过")


def test_circle_on_cell_border():
    test("圆形 r=0.5 — 圆心在 cell 边界上")
    g = make_grid(20, 20, {(5, 5)})
    check(is_circle_passable(g, 5.5, 5.0, R) == False,
          "圆心在 cell 边界, 邻格中心距离=0.5 → 碰撞")


def test_circle_vehicle_near_wall():
    test("圆形 r=0.5 — 车辆靠墙移动")
    g = make_grid(20, 20, {(4, 5)})
    # 圆心 (5,5), r=0.5 → 圆覆盖 [4.5, 5.5] × [4.5, 5.5]
    # cell(4,5) [4,5]×[5,6]: closest=(5,5), d=0 → 与圆重叠 → 碰撞!
    check(is_circle_passable(g, 5.0, 5.0, R) == False,
          "圆心 (5,5), 圆与邻格 cell(4,5) 墙重叠 → 碰撞")
    # 圆心 (6,5), r=0.5 → 圆覆盖 [5.5, 6.5]
    # cell(4,5) [4,5]: closest=(5,5), dx=1, d²=1 > 0.25 → 不重叠 → 安全
    check(is_circle_passable(g, 6.0, 5.0, R) == True,
          "圆心 (6,5), 圆离墙 1 格远 → 安全")


def test_circle_corridor():
    test("圆形 r=0.5 — 1格宽走廊")
    walls = {(3, y) for y in range(0, 20)} | {(5, y) for y in range(0, 20)}
    g = make_grid(20, 20, walls)
    # 圆心 (4,10) → 圆覆盖 [3.5, 4.5]，cell(3,10) [3,4] 重叠 → 撞墙
    check(is_circle_passable(g, 4.0, 10.0, R) == False,
          "圆心 (4,10) → 圆与 cell(3,10) 墙重叠 → 碰撞")
    # 圆心 (4.5,10) → 圆覆盖 [4.0, 5.0]，仅 cell(4,10) 重叠 → 安全
    check(is_circle_passable(g, 4.5, 10.0, R) == True,
          "圆心 (4.5,10) → 圆仅覆盖 cell(4,10) → 通过")


def test_circle_vertex():
    test("圆形 r=0.5 — 圆心在 4 格顶点，邻格有墙")
    g = make_grid(20, 20, {(5, 5)})
    # 圆心 (5.5, 5.5), r=0.5 → 圆覆盖 [5.0, 6.0]²
    # cell(5,5) [5,6]²: closest=(5.5,5.5), d=0 → 重叠墙!
    check(is_circle_passable(g, 5.5, 5.5, R) == False,
          "圆心在顶点，圆与 cell(5,5) 墙重叠 → 碰撞")


# ═══════════════════════════════════════════════════════
#  航向角 (Yaw) + 碰撞联动 测试
# ═══════════════════════════════════════════════════════

class VehicleState:
    """模拟车辆状态，带航向角"""
    def __init__(self, x: float, y: float, yaw: float = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw  # 弧度, 0=东(+x)

    def move_forward(self, distance: float) -> tuple:
        """沿 yaw 方向前进 distance 米，返回新坐标"""
        new_x = self.x + distance * math.cos(self.yaw)
        new_y = self.y + distance * math.sin(self.yaw)
        return (new_x, new_y)

    def try_move_forward(self, grid: MapGrid, distance: float) -> bool:
        """尝试前进，碰撞检测通过才更新 pose"""
        new_x, new_y = self.move_forward(distance)
        if is_circle_passable(grid, new_x, new_y, R):
            self.x, self.y = new_x, new_y
            return True
        return False


def test_yaw_east_clear():
    test("航向角 — yaw=0 (东), 前方无障碍")
    g = make_grid(20, 20, set())
    v = VehicleState(5.0, 5.0, yaw=0.0)
    ok = v.try_move_forward(g, 1.0)
    check(ok == True, "前进 1m → 通过")
    check(abs(v.x - 6.0) < 0.01 and abs(v.y - 5.0) < 0.01,
          f"新位置 ({v.x:.1f},{v.y:.1f}) ≈ (6.0,5.0)")


def test_yaw_diagonal():
    test("航向角 — yaw=π/4 (东北)")
    g = make_grid(20, 20, set())
    v = VehicleState(5.0, 5.0, yaw=math.pi / 4)
    ok = v.try_move_forward(g, 1.0)
    check(ok == True, "向东北前进 → 通过")
    ex = 5.0 + math.cos(math.pi / 4)
    ey = 5.0 + math.sin(math.pi / 4)
    check(abs(v.x - ex) < 0.01 and abs(v.y - ey) < 0.01,
          f"≈ ({ex:.2f},{ey:.2f})")


def test_yaw_toward_wall_blocked():
    test("航向角 — 前方是墙，移动被拒绝")
    g = make_grid(20, 20, {(6, 5)})
    v = VehicleState(5.0, 5.0, yaw=0.0)
    ok = v.try_move_forward(g, 1.0)
    check(ok == False, "目标 cell(6,5) 是墙 → 拒绝")
    check(v.x == 5.0 and v.y == 5.0,
          "位置未变 → 停在原地")


def test_yaw_parallel_to_wall():
    test("航向角 — 沿墙平行移动")
    g = make_grid(20, 20, {(6, y) for y in range(0, 20)})
    v = VehicleState(5.0, 5.0, yaw=math.pi / 2)  # 向北
    ok = v.try_move_forward(g, 1.0)
    check(ok == True, "向北移动，与墙平行 → 通过")
    check(abs(v.y - 6.0) < 0.01, f"y → {v.y:.1f} ≈ 6.0")


# ═══════════════════════════════════════════════════════
#  执行
# ═══════════════════════════════════════════════════════

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  碰撞检测测试套件 (r=0.5, 圆形, 航向角)")
    print("=" * 60)

    # MapGrid (8 组)
    test_mapgrid_basic()
    test_mapgrid_wall_check()
    test_mapgrid_bounds()
    test_mapgrid_from_voxels()
    test_mapgrid_from_wall_set()
    test_mapgrid_empty_voxels()
    test_mapgrid_oob_error()
    test_mapgrid_large()

    # Raycast (9 组)
    test_raycast_horizontal_clear()
    test_raycast_vertical_clear()
    test_raycast_diagonal_clear()
    test_raycast_wall_at_endpoint()
    test_raycast_wall_in_middle()
    test_raycast_wall_immediate()
    test_raycast_single_cell()
    test_raycast_negative_direction()
    test_raycast_grazing()

    # Circle r=0.5 (9 组)
    test_circle_open_center()
    test_circle_on_wall()
    test_circle_beside_wall_safe()
    test_circle_near_boundary_collision()
    test_circle_near_boundary_safe()
    test_circle_on_cell_border()
    test_circle_vehicle_near_wall()
    test_circle_corridor()
    test_circle_vertex()

    # Yaw (4 组)
    test_yaw_east_clear()
    test_yaw_diagonal()
    test_yaw_toward_wall_blocked()
    test_yaw_parallel_to_wall()

    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"  结果: {PASS}/{total} 通过", end="")
    if FAIL > 0:
        print(f", {FAIL} 失败 ❌")
    else:
        print(" ✅")
    print(f"{'='*60}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
