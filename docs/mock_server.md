# Mock Server

## 概述

Mock Server 是一个用 Python 编写的 WebSocket 服务端，模拟小车端 Pleiades 的行为，用于 Pictor 项目的端到端测试。它生成带障碍物的 2D 栅格地图，持续发送车辆位姿，并支持碰撞检测——车辆移动时检查是否会撞墙。

```
┌──────────────────────┐         WebSocket          ┌──────────────────────┐
│     Pictor (Godot)   │ ←─────────────────────────→ │   Mock Server (Python)│
│   WebSocketClient    │    map_full / pose / cmd    │   mock_vehicle.py     │
└──────────────────────┘                             └──────────────────────┘
```

## 基本信息

| 项目 | 值 |
|------|------|
| 语言 | Python 3.10+ |
| 依赖 | `websockets` (仅 mock_vehicle.py 需要), 标准库 |
| 默认端口 | 9090 |
| 地图大小 | 256×256 格（可扩展） |
| 每格含义 | 0 = 可通行 (free), 1 = 不可通行 (wall) |

---

## 文件结构

```
test_tool/
├── mock_vehicle.py        ← 主入口，WebSocket Server，生成地图 + 发送 pose
├── mock_map_grid.py       ← MapGrid 类，2D 栅格地图数据结构
├── mock_collision.py      ← 碰撞检测函数（Bresenham 线段 + 圆形碰撞）
└── test_collision.py      ← 碰撞检测测试套件（25 组, 52 条断言）
```

---

## 模块架构

```
mock_vehicle.py
  │
  ├── 地图生成: 256×256 随机 voxel 数组 (5% 墙密度)
  │       │
  │       ▼
  │   发送 map_full ──────────→ Pictor
  │
  ├── 位姿更新: 每秒递增 x 坐标
  │       │
  │       ▼
  │   发送 pose    ──────────→ Pictor
  │
  └── [未来] 碰撞检测集成:
          │
          ├── mock_map_grid.py
          │     MapGrid.from_voxels()
          │     MapGrid.is_passable(x, y)
          │
          └── mock_collision.py
                raycast(grid, x1, y1, x2, y2)
                is_circle_passable(grid, cx, cy, radius)
```

---

## 通信协议

Mock Server 遵循 [WebSocket 通信协议](websocket_protocol.md)，当前支持的消息：

### 上行 (Mock Server → Pictor)

| 消息类型 | 触发时机 | 内容 |
|---------|---------|------|
| `map_full` | 客户端连接后立即发送 | 256×256 全量 voxel 数组 |
| `pose` | 每秒发送一次 | 车辆位置 (x, y)、朝向 (yaw)、速度 (vx, vy) |
| `map_delta` | 暂未实现 | — |
| `path` | 暂未实现 | — |

### 下行 (Pictor → Mock Server)

| 消息类型 | 说明 | 状态 |
|---------|------|------|
| `cmd` | 控制命令 (forward/backward/spin/stop) | 暂未处理 |

---

## 碰撞检测

### 设计动机

Mock Server 的核心任务是模拟真实小车的行为。真实小车不会穿墙——当检测到前方有障碍物时，移动指令应当失效。碰撞检测模块为此提供两个维度的检查：

1. **线段碰撞**: 从当前位置到目标位置，直线路径上是否有墙？
2. **圆形碰撞**: 车辆作为一个圆形区域，是否与墙有重叠？

### MapGrid — 栅格地图

2D 占据栅格，使用 `bytearray` 一维数组存储，索引公式 `y * width + x`。

```
┌──┬──┬──┬──┬──┐
│0 │0 │1 │0 │0 │    cells = [0,0,1,0,0, 0,0,0,1,0, ...]
├──┼──┼──┼──┼──┤
│0 │0 │0 │1 │0 │    get_cell(x, y)  →  cells[y * width + x]
├──┼──┼──┼──┼──┤    set_cell(x, y, v) →  cells[y * width + x] = v
│0 │0 │0 │0 │0 │
└──┴──┴──┴──┴──┘

操作复杂度: O(1) — 纯数组下标访问
内存占用: width × height 字节 (256² = 64 KB)
```

核心 API:

| 方法 | 说明 |
|------|------|
| `get_cell(x, y)` | 返回格子值 |
| `set_cell(x, y, state)` | 设置格子值 |
| `is_wall(x, y)` | 是否为墙 (state == 1) |
| `is_passable(x, y)` | 是否可通行（在界内且非墙） |
| `in_bounds(x, y)` | 是否在边界内 |
| `from_voxels(voxels)` | 从 voxel 数组构造（自动推导尺寸） |
| `from_wall_set(w, h, walls)` | 从墙坐标集合构造（测试用） |

### Bresenham 线段碰撞

用于检查一段直线路径是否穿过墙壁。

```
原理: Bresenham 直线光栅化 + 逐格碰撞检查

  A(0,0) → B(4,3)
     y
     ↑
   3 ┌──┬──┬──┬──┬ B│
   2 │  │  │  │██│  │   沿直线逐格采样:
   1 │  │  │██│  │  │    (0,0)→(1,0)→(2,1)→(3,2)→(4,3)
   0 │A │  │  │  │  │                ↑ 撞墙!
     └──┴──┴──┴──┴──┘ → x
       0  1  2  3  4

算法: 增量误差累积器 d
  每步:
    向右走一格 → d += dy
    d ≥ dx 时  → d -= dx, 向上走一格
    检查当前格子 → is_wall? → 碰撞!
```

复杂度: O(max(|dx|, |dy|))，256² 地图最坏情况遍历 ≈ 362 格，耗时 < 0.1ms。

### 圆形碰撞检测

车辆简化为圆形，检查圆形区域是否与墙壁重叠。

```
圆形车辆碰撞检查:

  圆心: (cx, cy)，半径: r
  
  只遍历包围盒 [cx-r, cx+r] × [cy-r, cy+r]:
    for (gx, gy) in bounding_box:
        if (gx-cx)² + (gy-cy)² > r²:  ← 不在圆内，跳过
            continue
        if grid.is_wall(gx, gy):       ← 在圆内且是墙
            return False               ← 💥 碰撞!
        if not grid.in_bounds(gx, gy): ← 在圆内但出界
            return False               ← 💥 越界!
    return True                        ← 安全

  边界处理: d² > r² 跳过（严格内部），
            d² == r²（恰好在圆边界上）也算碰撞，作为安全余量。

  ┌──┬──┬──┬──┬──┐
  │  │  │░░│  │  │
  │  │░░│░░│░░│  │   圆心 (2,2), r=1.5
  │  │░░│🚗│░░│  │   检查 9 个格子，其中 4 个在圆内
  │  │  │░░│  │  │   (1,1)(2,1)(1,2)(2,2)(2,3)(3,2)
  │  │  │  │  │  │
  └──┴──┴──┴──┴──┘
```

---

## 碰撞行为规则

当车辆尝试移动到有障碍物的位置时：

```
收到移动指令
  │
  ├─ 计算目标位置 (x', y', yaw')
  ├─ is_circle_passable(grid, x', y', vehicle_radius)
  │
  ├─ True  → 更新 pose, 发送 pose_received
  └─ False → 放弃移动, pose 不变, 不发送消息
             (log: [⚠] collision at (x,y), move rejected)
```

- ❌ 不沿墙滑行
- ❌ 不重新寻路绕行
- ❌ 不反弹
- ❌ 不发送错误消息
- ✅ 停在原地，本次指令无效

---

## 测试

### 运行

```bash
cd test_tool && python test_collision.py
```

### 覆盖范围

| 模块 | 测试组 | 断言数 | 覆盖场景 |
|------|--------|--------|---------|
| MapGrid | 8 | 25 | get/set、is_wall、in_bounds、from_voxels、from_wall_set、空数组、越界、大规模性能 |
| raycast | 9 | 19 | 水平/垂直/对角线通行、终点墙、中途墙、起点墙、起点=终点、负方向、擦边 |
| is_circle_passable | 8 | 16 | 空旷、圆心有墙、边界碰撞、圆内墙、越界、r=0、浮点圆心、大半径 |
| **合计** | **25** | **52** | — |

### 性能

- 10000 次 `set_cell` 耗时 ≈ 3ms
- 1000×1000 网格 A* 寻路 ≈ 50ms（预估）

---

## 当前状态

| 功能 | 状态 |
|------|------|
| 地图生成 (map_full) | ✅ 已完成 |
| 位姿发送 (pose) | ✅ 已完成 |
| MapGrid 数据结构 | ✅ 已完成 |
| Bresenham 线段碰撞 | ✅ 已完成 |
| 圆形碰撞检测 | ✅ 已完成 |
| 碰撞检测测试 | ✅ 52/52 通过 |
| 碰撞检测集成到 mock_vehicle.py | ⬜ 待集成 |
| 路径规划 (A*) | ⏸️ 暂不实现 |
| cmd 命令接收处理 | ⏸️ 暂不实现 |
| 多 Chunk / 多车辆 | ⏸️ 暂不实现 |
