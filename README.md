# MockVehicle2D

2D 车辆模拟器，Python 实现。用于 Pictor 项目的 WebSocket 协议测试与碰撞检测验证。

## 快速开始

```bash
# 安装依赖
pip install pygame websockets

# 运行测试
python test_collision.py

# 启动 WebSocket Mock Server
python mock_vehicle.py

# 启动 Pygame 可视化
python mock_visual.py
```

## 文件结构

```
MockVehicle2D/
├── mock_map_grid.py       ← MapGrid 类，2D 栅格地图 (bytearray, O(1))
├── mock_collision.py      ← 碰撞检测：Bresenham 线段 + AABB vs Circle
├── mock_vehicle.py        ← WebSocket Server，模拟小车发送 map_full / pose
├── mock_visual.py         ← Pygame 可视化，W/S/A/D 驾驶，实时碰撞反馈
├── test_collision.py      ← 测试套件，60 条断言全部通过
└── docs/
    ├── mock_server.md     ← 本文档
    └── pygame_visual.md   ← Pygame 可视化设计文档
```

## 车辆参数

| 参数 | 值 |
|------|------|
| 形状 | 圆形 |
| 半径 | 0.5 (直径 = 1 cell = 1m) |
| 航向角 | yaw (弧度), 0 = +x |

## 碰撞检测

```
MapGrid (bytearray)
  cells[y * w + x]: 0 = 可通行, 1 = 墙
  操作: O(1)

Bresenham 线段碰撞
  逐格采样，任一格子是墙 → 碰撞
  复杂度: O(max(dx, dy))

AABB vs Circle 圆形碰撞
  最近点 = (clamp(cx, gx, gx+1), clamp(cy, gy, gy+1))
  重叠 ⇔ 距离² < r²
```

## 测试

```bash
python test_collision.py
```

| 模块 | 测试组 | 断言数 |
|------|--------|--------|
| MapGrid | 8 | 25 |
| raycast | 9 | 19 |
| is_circle_passable (r=0.5) | 9 | 16 |

## 通信协议

遵循 [WebSocket 通信协议](docs/websocket_protocol.md)。

| 方向 | 消息 | 状态 |
|------|------|------|
| 上行 (Server→Pictor) | `map_full` | ✅ |
| 上行 | `pose` | ✅ |
| 上行 | `map_delta` | ⏸️ |
| 下行 (Pictor→Server) | `cmd` | ⏸️ |
