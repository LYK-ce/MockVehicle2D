# MockVehicle2D

2D 车辆模拟器，Python 实现。用于 Pictor 项目的 WebSocket 协议测试与碰撞检测验证。

## 快速开始

```bash
# 一键安装（创建 .venv + 安装依赖）
bash bootstrap.sh

# 然后激活环境
source .venv/bin/activate

# 运行测试
mockvehicle2d test

# 启动可控 WebSocket Mock Server
mockvehicle2d serve --vehicle-id mock_vehicle_01

# 9090 被占用时改用其他端口，并在 Pictor 中输入相同端口
mockvehicle2d serve --port 19090 --vehicle-id mock_vehicle_01

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
│   ├── map_grid.py         ← MapGrid 类，可通行/墙体/无地面三态栅格
│   ├── collision.py        ← 碰撞检测：Bresenham 线段 + AABB vs Circle
│   ├── vehicle.py          ← Server/Pygame 共用的运动、碰撞与指令看门狗
│   ├── navigation.py       ← local odom 直达目标控制与状态
│   ├── safety.py           ← Tmini/边缘观测、固定阈值与本地安全运行时
│   ├── server.py           ← WebSocket Server，接收 cmd 并发送 map_full / pose / scan
│   ├── scan.py             ← YDLidar Tmini 二维角度/距离/强度扫描
│   └── visual.py           ← Pygame 可视化，支持 W+D 等组合驾驶与实时碰撞反馈
├── tests/
│   ├── test_collision.py   ← 碰撞检测测试套件
│   ├── test_scan.py        ← 二维扫描几何测试
│   ├── test_vehicle.py     ← 指令、运动、看门狗和防穿墙测试
│   ├── test_goto.py        ← goto 协议、状态、接管和碰撞测试
│   ├── test_safety.py      ← 纯安全感知与策略测试
│   ├── test_safety_runtime.py ← 自动/手动安全运行时接入测试
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
  cells[y * w + x]: 0 = 可通行, 1 = 墙, 2 = 无地面/落差
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

启动 Server 后，在 Pictor 中连接 `ws://127.0.0.1:9090`；使用 `--port 19090` 时，
Pictor 也应连接 `ws://127.0.0.1:19090`。连接首帧固定为
`{"type":"hello","vehicle_id":"mock_vehicle_01"}`，随后依次发送 `map_full → pose → scan`。

| 方向 | 消息 | 状态 |
|------|------|------|
| 上行 (Server→Pictor) | `hello` | ✅ |
| 上行 (Server→Pictor) | `map_full` | ✅ |
| 上行 | `pose` | ✅ |
| 上行 | `scan` | ✅ |
| 上行 | `map_delta` | ⏸️ |
| 下行 (Pictor→Server) | `cmd` / `drive` / `goto` | ✅ |

`scan` 默认使用 Tmini 轮廓：360°、0.02–12 m、名义 4000 Hz 测距、名义 6 Hz 扫描、667 条均匀射线。有效回波按 0.01 m 量化；无回波为 `range: 0.0, intensity: 0.0`，不能当作障碍物。水平 Tmini 只返回墙体；确定性的 `state=2` 落差测试区会显示在 `map_full`，由模拟的向下地面探测输入负责安全判断。

控制器可发送离散命令 `{"type":"cmd","seq":1,"cmd":"forward"}`，也可发送连续速度 `{"type":"drive","seq":2,"linear_mps":0.25,"angular_rps":-0.4}`；Server 都立即返回 `cmd_ack`。`drive` 的绝对值上限分别由 `--linear-speed` 和 `--angular-speed` 配置。超过 `--command-timeout` 未收到有效非零命令、收到非法命令或连接断开时，车辆自动停止；碰撞时停在最后一个安全位置。旧 `cmd` 格式保持兼容并与 `drive` 使用同一运动、碰撞和看门狗逻辑。

发送 `{"type":"goto","seq":3,"x_m":12.0,"y_m":8.5}` 可让模拟车在本地 `odom` 坐标中直达目标，Server 返回 `goto_ack`，并在 `pose.control_mode` 与 `pose.navigation` 中持续报告模式、状态、目标和结束原因。当前定位输入是 `simulator_ground_truth`；控制器只会先转向、直线前进和接近目标减速，不会规划绕行。自动行驶在障碍/边缘净空 `0.25–1.0 m` 内线性降速，净空 `<=0.25 m` 或安全输入故障时停车并报告 `blocked`；停止后不会自行恢复。任何手动 `cmd`/`drive` 或非法输入也会取消活动目标。

`pose.safety` 持续报告 `{state, reason, obstacle_clearance_m, edge_clearance_m}`。手动驾驶不在慢速区降速，但仍执行硬停止和故障停车；新的安全方向命令可解除手动安全锁停，纯旋转允许用于脱困。Tmini 只负责正障碍距离，落差净空是模拟的辅助下视/相机输入，不能解释为雷达能力。

`map_full` 与 `pose` 标有 `source: "simulator_ground_truth"`，仅供仿真验收和可视化；只有 `scan` 是模拟的 Tmini 本地观测，边缘输入也是模拟辅助量。未来真实导航不能把真值消息当作传感器输入。当前只有带本地安全门控的直达目标控制，没有路径规划、真实相机/下视传感器、真实定位、传感器噪声或 `map_delta`。
