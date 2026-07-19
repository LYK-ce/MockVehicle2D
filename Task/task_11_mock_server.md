# Task 11: Mock Server 完善

## 目标

完善 `test_tool/mock_vehicle.py`，使其成为一个功能完整的 Mock Server，能够模拟小车在带障碍物的地图上进行智能移动，用于 Pictor 项目的端到端测试。

## 方案决策

- **语言**: Python（非 Godot 原生）
- **理由**: 轻量、灵活、CI 友好、大规模地图性能优于 Godot 内置物理；碰撞检测/A* 纯 Python 手写 ~200 行，一次性投入
- **无第三方依赖**: 仅用标准库 (`asyncio`, `heapq`, `json`, `random`, `math`)

## 模块架构

```
test_tool/
├── mock_vehicle.py           ← 主入口，WebSocket server
├── mock_map_grid.py          ← Grid 数据结构 + 点查询
├── mock_collision.py         ← Bresenham 线段碰撞检测
└── mock_pathfinding.py       ← A* 寻路 + 随机漫步避障
```

## 功能清单

### Phase 1: 碰撞检测

- [ ] `MapGrid` 类 — 加载 voxel 数据，点查询 O(1)
- [ ] `check_point(x, y)` — 点是否合法（在界内且非墙）
- [ ] `raycast(x1, y1, x2, y2)` — Bresenham 线段碰撞，返回是否穿墙
- [ ] `validate_path(points)` — 整条路径的碰撞检查
- [ ] 碰撞响应：车辆撞墙后停止或沿墙滑动

### Phase 2: 路径规划

- [ ] `a_star(start, goal)` — 基于 heapq 的 A* 实现
- [ ] `random_walk(vehicle_pos, steps)` — 随机漫步，避免撞墙
- [ ] `waypoint_navigation(vehicle_pos, waypoints)` — 按航点序列移动

### Phase 3: 双向通信

- [ ] 接收 `ctrl_send` 命令（forward/backward/spin_left/spin_right/stop）
- [ ] 命令执行前做碰撞检测，拒绝撞墙命令
- [ ] `map_delta` 增量发送（车辆走到新 chunk 时）
- [ ] `path` 消息发送（车辆规划路径）
---

## Phase 1 详细设计: 碰撞检测

### MapGrid 数据结构
### Phase 2: 路径规划（暂不实现）

> ⏸️ 留待后续需求

### Phase 3: 双向通信（暂不实现）

> ⏸️ 留待后续需求

### Phase 4: 多车辆 / 大规模（暂不实现）

> ⏸️ 留待后续需求
```python
class MapGrid:
    """2D 占据栅格地图"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # 一维 array, index = y * width + x
        # 0 = 可通行, 1 = 墙
        self._cells = bytearray(width * height)

    def set_cell(self, x: int, y: int, state: int) -> None:
        self._cells[y * self.width + x] = state

    def get_cell(self, x: int, y: int) -> int:
        return self._cells[y * self.width + x]

    def is_wall(self, x: int, y: int) -> bool:
        return self.get_cell(x, y) == 1

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not self.is_wall(x, y)

    @classmethod
    def from_voxels(cls, voxels: list[dict]) -> "MapGrid":
        """从 voxel 数组构造 — 自动推导 size"""
        ...
```

### 点碰撞检查

```
输入: 车辆下一个位置 (x, y)
过程: grid.is_passable(x, y)
输出: True(可通行) / False(撞墙/出界)

复杂度: O(1) — 纯数组下标访问
```

### 线段碰撞检查 (Bresenham)

```
输入: 起点 A(x1,y1) → 终点 B(x2,y2)
过程:
  1. Bresenham 直线算法, 逐格遍历线段经过的所有格子
  2. 对每个格子调用 grid.is_passable()
  3. 遇到第一个墙 → 返回 (True, 碰撞点坐标)
  4. 全部通过 → 返回 (False, None)

输出: (has_collision: bool, collision_point: tuple | None)

复杂度: O(max(|dx|, |dy|))
256×256 地图最坏情况: 遍历 ~362 个格子, < 0.1ms
```

### 车辆碰撞模型

车辆简化为圆形占用区域:

```
圆形碰撞体
  - 半径 r, 检查以 (x,y) 为中心、r 为半径的圆形区域内所有格子
  - 格子 (cx,cy) 碰撞 ⇔ (cx-x)² + (cy-y)² < r²
  - 优化: 只遍历包围盒 [x-r, x+r] × [y-r, y+r]
```

> 已确定使用圆形模型。点车辆（方案A）和 OBB（方案C）不再考虑。

### 碰撞行为规则

**唯一规则**: 碰撞 → 本次移动失效，车辆停在原地不动。

```
车辆收到移动指令（forward/backward/spin_left/spin_right）
  │
  ├─ 1. 根据当前 pose + cmd 计算出目标位置 (x', y', yaw')
  │
  ├─ 2. 圆形碰撞检测:
  │      is_circle_passable(grid, x', y', vehicle_radius)
  │
  ├─ 3a. 通过 → 更新 pose, 发送 pose_received
  │
  └─ 3b. 碰撞 → 放弃移动, pose 不变, 不发送消息
                    （可在 log 中输出 [⚠] collision at (x',y'), move rejected）
```

**不做的行为**（明确排除）:
- ❌ 不沿墙滑行
- ❌ 不重新寻路绕行
- ❌ 不反弹
- ❌ 不发送错误消息给 client
- ✅ 就是停在原地，本次指令无效

---

## 依赖

- 无外部依赖（仅 Python 3.10+ 标准库）

## 相关文件

| 文件 | 说明 |
|------|------|
| `test_tool/mock_vehicle.py` | 现有 mock server，需重构 |
| `src/utils/coords.gd` | 坐标转换参考 |
| `src/renderer_2d/map_data_2d.gd` | Chunk 数据结构参考 |
| `Architecture/architecture.md` | 协议格式参考 |

## 状态

- [x] Phase 1: 碰撞检测 — 已完成 (2026-07-13)
- [x] 测试用例编写 — 52/52 通过
- [⏸️] Phase 2: 路径规划（暂不实现）
- [⏸️] Phase 3: 双向通信（暂不实现）
- [⏸️] Phase 4: 多车辆 / 大规模（暂不实现）

## 交付物

| 文件 | 说明 |
|------|------|
| `test_tool/mock_map_grid.py` | MapGrid 类 — 2D 栅格地图 (bytearray) |
| `test_tool/mock_collision.py` | 碰撞检测 — Bresenham 线段 + 圆形碰撞 |
| `test_tool/test_collision.py` | 测试套件 — 25 个测试组, 52 条断言 |
