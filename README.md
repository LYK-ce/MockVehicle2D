# MockVehicle2D

2D 车辆模拟器，Python 实现。用于 Pictor 项目的 WebSocket 协议测试与碰撞检测验证。

## 快速开始

```bash
# 一键安装（创建 .venv + 安装依赖）
source bootstrap.sh

# 然后激活环境
source .venv/bin/activate

# 运行测试
mockvehicle2d test

# 启动可控 WebSocket Mock Server
mockvehicle2d serve --vehicle-id mock_vehicle_01

# 校准模拟车（角速度单位为度/秒）
mockvehicle2d serve --linear-speed 0.5 --angular-speed 90 --vehicle-radius 0.5 --command-timeout 1.0

# 启动 Pygame 可视化
mockvehicle2d visual

# 或通过 Python 模块运行
python -m mockvehicle2d test
```

> **注意**：依赖安装在项目本地的 `.venv/` 中，不会污染系统 Python。

## 文件结构

```
MockVehicle2D/
├── src/mockvehicle2d/
│   ├── cli.py              ← 统一 CLI 入口 (argparse)
│   ├── map_grid.py         ← MapGrid 类，2D 栅格地图 (bytearray, O(1))
│   ├── collision.py        ← 碰撞检测：Bresenham 线段 + AABB vs Circle
│   ├── vehicle.py          ← Server/Pygame 共用的运动、碰撞与指令看门狗
│   ├── server.py           ← WebSocket Server，接收 cmd 并发送 map_full / pose / scan
│   ├── scan.py             ← YDLidar Tmini 二维角度/距离/强度扫描
│   └── visual.py           ← Pygame 可视化，W/S/A/D 驾驶，实时碰撞反馈
├── tests/
│   ├── test_collision.py   ← 碰撞检测测试套件
│   ├── test_scan.py        ← 二维扫描几何测试
│   ├── test_vehicle.py     ← 指令、运动、看门狗和防穿墙测试
│   └── test_server_scan.py ← scan WebSocket 帧测试
├── docs/
│   ├── mock_server.md
│   └── pygame_visual.md
├── pyproject.toml
├── LICENSE
└── README.md
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
mockvehicle2d test
```

| 模块 | 测试组 | 断言数 |
|------|--------|--------|
| MapGrid | 8 | 25 |
| raycast | 9 | 19 |
| is_circle_passable (r=0.5) | 9 | 16 |

## 通信协议

遵循 [WebSocket 通信协议](docs/websocket_protocol.md)。

启动 Server 后，在 Pictor 中连接 `ws://127.0.0.1:9090`。连接首帧固定为
`{"type":"hello","vehicle_id":"mock_vehicle_01"}`，随后依次发送 `map_full → pose → scan`。

| 方向 | 消息 | 状态 |
|------|------|------|
| 上行 (Server→Pictor) | `hello` | ✅ |
| 上行 (Server→Pictor) | `map_full` | ✅ |
| 上行 | `pose` | ✅ |
| 上行 | `scan` | ✅ |
| 上行 | `map_delta` | ⏸️ |
| 下行 (Pictor→Server) | `cmd` | ✅ |

`scan` 默认使用 Tmini 轮廓：360°、0.02–12 m、名义 4000 Hz 测距、名义 6 Hz 扫描、667 条均匀射线。有效回波按 0.01 m 量化；无回波为 `range: 0.0, intensity: 0.0`，不能当作障碍物。

控制器发送 `{"type":"cmd","seq":1,"cmd":"forward"}` 后，Server 立即返回 `cmd_ack`，并从同一个车辆状态按顺序发送 `pose` 与 `scan`。超过 `--command-timeout` 未收到有效命令、收到非法命令或连接断开时，车辆自动停止；碰撞时停在最后一个安全位置。

`map_full` 与 `pose` 标有 `source: "simulator_ground_truth"`，仅供仿真验收和可视化；只有 `scan` 是模拟的 Tmini 本地观测。未来导航算法不能把真值消息当作真实传感器输入。当前没有实现寻路、相机、定位误差、雷达噪声或 `map_delta`。
