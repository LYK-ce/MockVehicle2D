# Mock Server — 架构与碰撞检测

## 概述

MockVehicle2D 模拟小车行为：生成 2D 栅格地图、发送位姿、响应控制命令。
通过 WebSocket 与 Pictor (Godot) 通信，并可选 Pygame 可视化窗口进行本地调试。

```
┌──────────────────────┐         WebSocket          ┌──────────────────────┐
│     Pictor (Godot)   │ ←─────────────────────────→ │   MockVehicle2D      │
│   WebSocketClient    │ map_full / pose / scan / cmd│   mock_vehicle.py    │
└──────────────────────┘                             └──────────────────────┘
```

## 基本信息

| 项目 | 值 |
|------|------|
| 语言 | Python 3.10+ |
| 依赖 | `websockets` (WebSocket), `pygame` (可视化), 标准库 |
| 默认端口 | 9090 |
| 地图规模 | 24×24 (pygame) / 256×256 (WebSocket) |
| 车辆 | 圆形, r=0.5, 有航向角 yaw |
| 每格含义 | 0 = 可通行, 1 = 不可通行 (wall) |

---

## 文件结构

```
MockVehicle2D/
├── src/mockvehicle2d/
│   ├── server.py           ← WebSocket Server 入口
│   ├── scan.py             ← YDLidar 风格二维激光扫描
│   ├── map_grid.py         ← MapGrid 栅格地图数据结构
│   ├── collision.py        ← Bresenham 线段 + AABB 圆形碰撞检测
│   └── visual.py           ← Pygame 可视化测试
├── tests/
│   ├── test_collision.py   ← 碰撞检测测试 (60 条断言)
│   ├── test_scan.py        ← 二维扫描几何测试
│   └── test_server_scan.py ← scan WebSocket 帧测试
├── docs/
│   ├── mock_server.md      ← 本文档
│   └── pygame_visual.md    ← Pygame 可视化设计
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 模块架构

```
mockvehicle2d serve (WebSocket Server)
  │
  ├── 地图生成: voxel 数组 → map_full
  │
  ├── 位姿发送: 名义 6 Hz pose
  │
  ├── 本地激光: 名义 6 Hz Tmini scan（二维角度/距离/强度）
  │
  └── 碰撞检测 (可集成):
        ├── map_grid.py  → MapGrid
        └── collision.py → is_circle_passable / raycast

mockvehicle2d visual (Pygame 可视化)
  ├── import mockvehicle2d.map_grid
  ├── import mockvehicle2d.collision
  └── CharacterBody2D 模拟: W/S/A/D 驾驶 + 碰撞反馈
```

---

## 通信协议

遵循 [WebSocket 通信协议](websocket_protocol.md)。

| 方向 | 消息 | 状态 |
|------|------|------|
| 上行 (Server → Pictor) | `map_full` | ✅ 已实现 |
| 上行 | `pose` | ✅ 已实现 |
| 上行 | `scan` | ✅ 已实现 |
| 上行 | `map_delta` | ⏸️ 暂不实现 |
| 下行 (Pictor → Server) | `cmd` | ⏸️ 暂不实现 |

---

## 碰撞检测

### MapGrid — 栅格地图

2D 占据栅格，`bytearray` 一维数组，索引 `y * width + x`。

```
┌──┬──┬──┬──┬──┐
│0 │0 │1 │0 │0 │
├──┼──┼──┼──┼──┤
│0 │0 │0 │1 │0 │    get_cell(x, y)  → cells[y * width + x]
├──┼──┼──┼──┼──┤    set_cell(x, y, v) → cells[y * width + x] = v
│0 │0 │0 │0 │0 │
└──┴──┴──┴──┴──┘

O(1) 查询, O(1) 写入, 256² = 64 KB
```

### Bresenham 线段碰撞

直线路径逐格采样，途中任一 cell 为墙 → 碰撞。

```
A(0,0) → B(4,3):  (0,0)→(1,0)→(2,1)→(3,2)→(4,3)
                                     ↑ 撞墙!
复杂度: O(max(|dx|, |dy|))
```

### AABB vs Circle 圆形碰撞

车辆为圆形 (r=0.5)，检测与 cell AABB 的重叠：

```
cell [gx, gx+1] × [gy, gy+1] 到圆心 (cx, cy) 的最近点:

  closest_x = clamp(cx, gx, gx+1)
  closest_y = clamp(cy, gy, gy+1)

  重叠 ⇔ (closest_x - cx)² + (closest_y - cy)² < r²
```

```
  ┌─────┬─────┐
  │     │  🚗 │  圆心 (5.3, 5.0), r=0.5
  │cell │cell │  cell(4,5): closest=(5.0, 5.0), d=0.3 → 重叠
  │(4,5)│(5,5)│  cell(5,5): closest=(5.3, 5.0), d=0 → 重叠
  └─────┴─────┘
```

---

## 碰撞行为

```
移动指令 → 计算目标位置 → is_circle_passable()
  ├─ True  → 更新 pose
  └─ False → 停在原地 (本次指令无效)
```

- ❌ 不滑行  ❌ 不绕行  ❌ 不反弹
- ✅ 停在原地

---

## 测试

```bash
mockvehicle2d test
```

| 模块 | 测试组 | 断言数 | 覆盖 |
|------|--------|--------|------|
| MapGrid | 8 | 25 | get/set, 越界, from_voxels, 性能 |
| raycast | 9 | 19 | 水平/垂直/对角, 撞墙, 擦边 |
| is_circle_passable | 9 | 16 | 圆心墙, 边界, 邻格重叠, 走廊, 航向角联动 |
| **合计** | **26** | **60** | — |

---

## 当前状态

| 功能 | 状态 |
|------|------|
| 地图生成 (map_full) | ✅ |
| 位姿发送 (pose) | ✅ |
| 本地二维激光 (scan) | ✅ |
| MapGrid 栅格地图 | ✅ |
| Bresenham 线段碰撞 | ✅ |
| AABB vs Circle 碰撞 | ✅ |
| 碰撞检测测试 | ✅ 60/60 |
| 二维扫描测试 | ✅ |
| Pygame 可视化 | ✅ |
| 路径规划 (A*) | ⏸️ |
| cmd 命令接收 | ⏸️ |
