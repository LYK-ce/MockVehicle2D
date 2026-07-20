# Mock Server — 架构与碰撞检测

## 概述

MockVehicle2D 模拟小车行为：生成 2D 栅格地图、发送位姿、响应控制命令。
通过 WebSocket 与 Pictor (Godot) 通信，并可选 Pygame 可视化窗口进行本地调试。

```
┌──────────────────────┐         WebSocket          ┌──────────────────────┐
│     Pictor (Godot)   │ ←─────────────────────────→ │   MockVehicle2D      │
│   WebSocketClient    │ hello / map / pose / scan / cmd│ MockVehicle2D       │
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
│   ├── vehicle.py          ← 共用车辆运动、碰撞与指令看门狗
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
  ├── 连接握手: hello(vehicle_id)
  ├── 地图生成: voxel 数组 → map_full
  │
  ├── cmd 接收: 严格校验 → cmd_ack / error → 故障停车
  │
  ├── 共用车辆状态: 实际单调时间 → 运动 / 看门狗 / 分步碰撞
  │
  ├── 同帧遥测: 名义 6 Hz pose → Tmini scan（相同 seq / ts）
  │
  └── 碰撞检测:
        ├── map_grid.py  → MapGrid
        └── collision.py → is_circle_passable / raycast

mockvehicle2d visual (Pygame 可视化)
  ├── import mockvehicle2d.map_grid
  ├── import mockvehicle2d.vehicle
  └── W/S/A/D/方向键驾驶 + 同一运动和碰撞逻辑
```

---

## 通信协议

遵循 [WebSocket 通信协议](websocket_protocol.md)。

| 方向 | 消息 | 状态 |
|------|------|------|
| 上行 (Server → Pictor) | `hello` | ✅ 已实现 |
| 上行 (Server → Pictor) | `map_full` | ✅ 已实现 |
| 上行 | `pose` | ✅ 已实现 |
| 上行 | `scan` | ✅ 已实现 |
| 上行 | `map_delta` | ⏸️ 暂不实现 |
| 下行 (Pictor → Server) | `cmd` | ✅ 已实现 |
| 上行 | `cmd_ack` / `error` | ✅ 已实现 |

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
移动指令 → 按实际 dt 切分小步 → is_swept_circle_passable()
  ├─ True  → 更新 pose 并继续
  └─ False → 停在最后安全点，command=stop, collision=true
```

- ❌ 不滑行  ❌ 不绕行  ❌ 不反弹
- ✅ 不穿墙并报告零实际速度

## 控制与故障停车

```bash
mockvehicle2d serve \
  --vehicle-id mock_vehicle_01 \
  --linear-speed 0.5 \
  --angular-speed 90 \
  --vehicle-radius 0.5 \
  --command-timeout 1.0
```

角速度参数单位为度/秒。规范命令为 `{"type":"cmd","seq":0,"cmd":"forward"}`；也兼容精确 legacy 格式 `{"cmd":"forward"}`。新命令生效前先把旧命令积分到接收时刻。非法消息、看门狗超时、碰撞或连接结束都会停车。

Pictor 连接 `ws://127.0.0.1:9090` 后，首帧为 `hello`，随后为 `map_full → pose → scan`。`hello` 不携带地址；Pictor 使用自己实际连接的 URL。

每个 6 Hz deadline 只推进一次共用状态，再以相同 `seq` 和 Unix `ts` 顺序发送 `pose`、`scan`。命令接收和全部发送都在同一个连接协程内串行执行；每轮先处理已到期遥测，命令洪泛不会永久饿死遥测。

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
| Pictor hello 握手 | ✅ |
| 地图生成 (map_full) | ✅ |
| 位姿发送 (pose) | ✅ |
| 本地二维激光 (scan) | ✅ |
| MapGrid 栅格地图 | ✅ |
| Bresenham 线段碰撞 | ✅ |
| AABB vs Circle 碰撞 | ✅ |
| 碰撞检测测试 | ✅ 60/60 |
| 二维扫描测试 | ✅ |
| Pygame 可视化 | ✅ |
| cmd 命令接收、确认和错误停车 | ✅ |
| 实际时间运动、看门狗和防穿墙 | ✅ |
| 寻路算法 | ⏸️ 暂不选型 |

`map_full` 和 `pose` 的 `source` 为 `simulator_ground_truth`，只用于仿真验收与可视化；只有 `scan` 是模拟的 Tmini 本地观测。当前没有实现任何寻路算法、相机、定位误差、雷达噪声或 `map_delta`。
