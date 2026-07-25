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

# 校准模拟车（公开接口统一使用 SI：m、s、rad）
mockvehicle2d serve --linear-speed-mps 0.5 --angular-speed-rps 1.5708 \
  --vehicle-radius-m 0.5 --command-timeout-s 1.0

# 配置出生锚点与可重放的里程计误差
mockvehicle2d serve --anchor-id car_01_anchor --anchor-x-m 10 --anchor-y-m 10 \
  --anchor-yaw-rad 0 --odom-translation-noise-m 0.01 \
  --odom-yaw-noise-rad 0.0035 --odom-seed 42

# A* 全真值调试工具（不属于 goto 运行链路）
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200

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
│   ├── pathfinding/        ← D* Lite 动态规划 + A* 全真值调试 + 路径跟随
│   │   ├── a_star.py
│   │   ├── d_star_lite.py
│   │   ├── waypoint_follower.py
│   │   └── __init__.py
│   ├── map_grid.py         ← MapGrid 类，2D 栅格地图 (bytearray, O(1))
│   ├── collision.py        ← 碰撞检测：Bresenham 线段 + AABB vs Circle
│   ├── vehicle.py          ← Server/Pygame 共用的运动、碰撞与指令看门狗
│   ├── local_state.py      ← 出生锚点、增量里程计与车辆自有观测地图
│   ├── navigation.py       ← 有限视野 D* Lite 重规划、局部目标跟踪与状态
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
│   ├── test_scan_matching.py ← 有限窗配准、拒绝门控与漂移修正测试
│   ├── test_d_star_lite.py ← 增量规划、动态变化和参考最短路对照
│   ├── test_dynamic_navigation.py ← 有限视野发现、重规划与到达 E2E
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

`goto` 只读取锚定局部位姿和 Tmini 累积出的有限视野 `ObservedGrid`。未知格可通行但
代价高于已确认自由格，且驶向未知格时降速；占用格按车辆半径膨胀后不可通行。每次扫描使
格子在 Unknown / Free / Occupied 间变化时，D* Lite 复用旧搜索状态增量修复路径。

```bash
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200
```

| 组件 | 文件 | 说明 |
|------|------|------|
| D* Lite | `pathfinding/d_star_lite.py` | 有限规划窗、八连通、Unknown 高代价、半径膨胀、增量重规划 |
| 自动导航 | `navigation.py` | 将局部地图路径编译为速度，未知区降速并服从安全门控 |
| A* 调试 | `pathfinding/a_star.py` | 只供 `pathfind` 在完整模拟真值地图上做离线参考 |
| Legacy 跟随器 | `pathfinding/waypoint_follower.py` | 兼容旧的预编译静态路径调用 |

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

`mockvehicle2d test` 使用当前 Python 解释器运行 `tests/` 下的完整 pytest
测试集，包括有限视野 D* Lite、动态重规划、scan matching、SI 契约、WebSocket
协议、车辆运动、碰撞与安全回归；任一测试失败时命令返回非零状态。

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

每辆车启动时只知道 `global_map → anchor_map` 的出生锚点。随后用运动增量生成
`anchor_map → odom → base_link → lidar` 的锚定里程计，并把 Tmini 扫描累计到独立的
`ObservedGrid`（`Unknown/Free/Occupied`）。该地图及 revision 在 Pictor 断开重连后仍保留，
但当前只存在小车内存中，不上传中央地图。每帧先用 bounded correlative scan matching
将当前扫描与旧的 Occupied 证据配准，通过支持数、得分和歧义 margin 后才修正里程计，再
写入当前帧。默认噪声为零以保持确定性；非零噪声使用 `--odom-seed` 重放。这是最小局部
SLAM 前端，不含回环检测、位姿图或全局优化，不能宣称生产级定位。

控制器可发送离散命令 `{"type":"cmd","seq":1,"cmd":"forward"}`，也可发送连续速度 `{"type":"drive","seq":2,"linear_mps":0.25,"angular_rps":-0.4}`；Server 都立即返回 `cmd_ack`。`drive` 的绝对值上限分别由 `--linear-speed-mps` 和 `--angular-speed-rps` 配置。超过 `--command-timeout-s` 未收到有效非零命令、收到非法命令或连接断开时，车辆自动停止；碰撞时停在最后一个安全位置。旧 `cmd` 格式保持兼容并与 `drive` 使用同一运动、碰撞和看门狗逻辑。

发送 `{"type":"goto","seq":3,"x_m":12.0,"y_m":8.5}` 可让模拟车自主前往
`global_map` 中的目标；Server 在锚点边界将其转换为 `anchor_map` 坐标。`pose` 顶层坐标仍
兼容 Pictor，但来源为 `anchored_odometry`，并附带局部位姿协方差、quality 和 revision。
控制器使用有限规划窗内的 D* Lite 路径；尚未观测的障碍只有进入 Tmini 视野、写入局部图后
才触发增量绕行。定位 `degraded` 时自动线速限制为一半；定位 `lost` 时自动任务停车并变为
`blocked`，且停止写入本地观测地图。自动
行驶在障碍/边缘净空 `0.25–1.0 m` 内线性降速，净空 `<=0.25 m` 或安全输入故障时停车；
任何手动 `cmd`/`drive` 或非法输入也会取消活动目标。

`pose.safety` 持续报告 `{state, reason, obstacle_clearance_m, edge_clearance_m}`。障碍净空按圆形车体沿行驶方向扫过的完整走廊计算，不使用固定角度扇区；运行时把延迟时段拆成不超过 `0.05 m` 且不越过硬停止净空的小步，每步重新观测。手动驾驶不在慢速区降速，但仍执行硬停止和故障停车；新的安全方向命令可解除手动安全锁停，纯旋转允许用于脱困。Tmini 只负责正障碍距离，落差净空是模拟的辅助下视/相机输入，不能解释为雷达能力。

`map_full` 仍标有 `source: "simulator_ground_truth"`，仅用于物理碰撞、生成传感器数据和
调试显示；正常 WebSocket `pose` 不再泄露绝对真值。模拟 scan 生成器仍必须读取真值环境，
但导航只消费锚定里程计、车辆自有地图和局部安全结果。无回波目前按最大量程更新为自由空间，这是当前
模拟协议约定，接入真实 Tmini 前必须按硬件无回波语义校准。水平 Tmini 不能检测跌落。
当前已实现有限视野局部栅格、轻量 scan matching 和 D* Lite；仍没有回环、全局优化、地图
持久化或中央地图同步。
