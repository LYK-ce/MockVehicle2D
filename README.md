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

# 启动可控 WebSocket Mock Server（默认端口 19090）
mockvehicle2d serve --vehicle-id mock_vehicle_01

# 端口被占用时改用其他端口，并在 Pictor 中输入相同端口
mockvehicle2d serve --port 9090 --vehicle-id mock_vehicle_01

# 校准模拟车（角速度单位为度/秒）
mockvehicle2d serve --linear-speed 0.5 --angular-speed 90 --vehicle-radius 0.5 --command-timeout 1.0

# 配置出生锚点与可重放的里程计误差
mockvehicle2d serve --anchor-id car_01_anchor --anchor-x 10 --anchor-y 10 --anchor-yaw 0 \
  --odom-translation-noise 0.01 --odom-yaw-noise 0.2 --odom-seed 42

# A* 寻路（在 256×256 随机地图上规划路径）
mockvehicle2d pathfind --start 10,10 --goal 200,200

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
│   ├── cli/                ← 统一 CLI 入口 (argparse)
│   │   └── main.py
│   ├── pathfinding/        ← A* 寻路 + 路径跟随
│   │   ├── a_star.py
│   │   ├── waypoint_follower.py
│   │   └── __init__.py
│   ├── map_grid.py         ← MapGrid 类，2D 栅格地图 (bytearray, O(1))
│   ├── collision.py        ← 碰撞检测：Bresenham 线段 + AABB vs Circle
│   ├── vehicle.py          ← Server/Pygame 共用的运动、碰撞与指令看门狗
│   ├── local_state.py      ← 出生锚点、增量里程计与车辆自有观测地图
│   ├── navigation.py       ← local odom 直达目标控制与状态
│   ├── safety.py           ← Tmini/边缘观测、固定阈值与本地安全运行时
│   ├── server.py           ← WebSocket Server，接收 cmd 并发送 map_full / pose / scan
│   ├── scan.py             ← YDLidar Tmini 二维角度/距离/强度扫描
│   └── visual.py           ← Pygame 可视化，支持 W+D 等组合驾驶与实时碰撞反馈
├── tests/
│   ├── test_collision.py   ← 碰撞检测测试套件
│   ├── test_pathfinding.py ← A* 寻路 + 路径跟随测试
│   ├── test_scan.py        ← 二维扫描几何测试
│   ├── test_vehicle.py     ← 指令、运动、看门狗和防穿墙测试
│   ├── test_goto.py        ← goto 协议、状态、接管和碰撞测试
│   ├── test_local_state.py ← 锚点、里程计、观测地图与定位质量测试
│   ├── test_safety.py      ← 纯安全感知与策略测试
│   ├── test_safety_runtime.py ← 自动/手动安全运行时接入测试
│   └── test_server_scan.py ← scan WebSocket 帧测试
├── docs/
│   ├── mock_server.md
│   ├── pathfinding.md
│   ├── pygame_visual.md
│   └── websocket_protocol.md
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

## 寻路

车辆可从任意起点自动规划避障路径并导航到终点。

```bash
mockvehicle2d pathfind --start 10,10 --goal 200,200
```

| 组件 | 文件 | 说明 |
|------|------|------|
| A* 搜索 | `pathfinding/a_star.py` | 八连通，欧几里得启发式，对角线剪枝，1-cell 膨胀 |
| 路径跟随 | `pathfinding/waypoint_follower.py` | 网格路径 → Vehicle cmd 序列，朝向跟踪 |

算法详情参见 [寻路文档](docs/pathfinding.md)。

## 碰撞检测

```
MapGrid (bytearray)
  cells[y * w + x]: 0 = 可通行, 1 = 墙, 2 = 无地面/落差
  操作: O(1)

Bresenham 线段碰撞
  逐格采样，任一格子不可通行（墙、落差或越界）→ 碰撞
  复杂度: O(max(dx, dy))

AABB vs Circle 圆形碰撞
  最近点 = (clamp(cx, gx, gx+1), clamp(cy, gy, gy+1))
  重叠 ⇔ 距离² < r²
```

## 测试

`mockvehicle2d test` 会运行栅格/碰撞、Tmini 扫描、车辆运动、WebSocket
协议、`goto`、安全策略与延迟执行回归测试。

## 通信协议

遵循 [WebSocket 通信协议](docs/websocket_protocol.md)。

启动 Server 后，在 Pictor 中连接 `ws://127.0.0.1:19090`；使用 `--port 9090` 时，
Pictor 也应连接 `ws://127.0.0.1:9090`。连接首帧固定为
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

每辆车启动时只知道 `global_map → anchor_map` 的出生锚点。随后用物理运动增量生成
`anchor_map → odom → base_link → lidar` 的锚定里程计，并把 Tmini 扫描累计到独立的
`ObservedGrid`（`Unknown/Free/Occupied`）。该地图及 revision 在 Pictor 断开重连后仍保留，
但当前只存在小车内存中，不上传中央地图。默认噪声为零以保持确定性；非零噪声使用
`--odom-seed` 重放。它不是 SLAM，无法修正长期漂移。

控制器可发送离散命令 `{"type":"cmd","seq":1,"cmd":"forward"}`，也可发送连续速度 `{"type":"drive","seq":2,"linear_mps":0.25,"angular_rps":-0.4}`；Server 都立即返回 `cmd_ack`。`drive` 的绝对值上限分别由 `--linear-speed` 和 `--angular-speed` 配置。超过 `--command-timeout` 未收到有效非零命令、收到非法命令或连接断开时，车辆自动停止；碰撞时停在最后一个安全位置。旧 `cmd` 格式保持兼容并与 `drive` 使用同一运动、碰撞和看门狗逻辑。

发送 `{"type":"goto","seq":3,"x_m":12.0,"y_m":8.5}` 可让模拟车直达
`global_map` 中的目标；Server 在锚点边界将其转换为 `anchor_map` 坐标。`pose` 顶层坐标仍
兼容 Pictor，但来源为 `anchored_odometry`，并附带局部位姿协方差、quality 和 revision。
控制器只会先转向、直线前进和接近目标减速，不会规划绕行。定位 `degraded` 时自动线速
限制为一半；定位 `lost` 时自动任务停车并变为 `blocked`，且停止写入本地观测地图。自动
行驶在障碍/边缘净空 `0.25–1.0 m` 内线性降速，净空 `<=0.25 m` 或安全输入故障时停车；
停止后不会自行恢复。任何手动 `cmd`/`drive` 或非法输入也会取消活动目标。

`pose.safety` 持续报告 `{state, reason, obstacle_clearance_m, edge_clearance_m}`。障碍净空按圆形车体沿行驶方向扫过的完整走廊计算，不使用固定角度扇区；运行时把延迟时段拆成不超过 `0.05 m` 且不越过硬停止净空的小步，每步重新观测。手动驾驶不在慢速区降速，但仍执行硬停止和故障停车；新的安全方向命令可解除手动安全锁停，纯旋转允许用于脱困。Tmini 只负责正障碍距离，落差净空是模拟的辅助下视/相机输入，不能解释为雷达能力。

`map_full` 仍标有 `source: "simulator_ground_truth"`，仅用于物理碰撞、生成传感器数据和
调试显示；正常 WebSocket `pose` 不再泄露绝对真值。模拟 scan 生成器仍必须读取真值环境，
但导航只消费锚定里程计和局部安全结果。无回波目前按最大量程更新为自由空间，这是当前
模拟协议约定，接入真实 Tmini 前必须按硬件无回波语义校准。水平 Tmini 不能检测跌落。
当前没有实现 SLAM、scan matching、回环、D* Lite 或中央地图同步。
