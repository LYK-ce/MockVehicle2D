# MockVehicle2D

MockVehicle2D 是面向真实小车控制栈的二维确定性模拟器。当前重点是验证
Robot Controller 的模式切换、任务队列、有限视野自主导航、本地安全和 WebSocket
协议；不实现中央地图系统，也不把仿真真值地图提供给自主规划器。

## 架构

```text
WebSocket JSON
      │
      ▼
严格协议边界（mode / manual / auto）
      │
      ▼
RobotController（唯一控制权）
  ├── Manual：速度设定值 + 看门狗
  └── Auto：任务队列 + Goto 执行器
                         │
                         ▼
Tmini scan + odometry → ObservedGrid → D* Lite → 速度设定值
                                                  │
                         Manual / Auto ───────────┤
                                                  ▼
                                      LocalSafetyRuntime
                                                  │
                                                  ▼
                                               Vehicle
```

`RobotController` 是唯一向车辆安装运动设定值的业务控制器。WebSocket Server 只负责
连接、协议解析、帧调度和遥测；`GotoController` 只计算期望速度；手动和自动速度都必须
经过同一个本地安全运行时。碰撞检测和安全运行时仍可直接执行独立的故障停车。

## 快速开始

```bash
bash bootstrap.sh
source .venv/bin/activate

python -m pytest

# 默认 ws://0.0.0.0:19090
mockvehicle2d serve --vehicle-id mock_vehicle_01

# 改端口，并配置任务队列、车辆和出生锚点
mockvehicle2d serve \
  --port 9090 \
  --mission-capacity 16 \
  --linear-speed-mps 0.5 \
  --angular-speed-rps 1.5708 \
  --vehicle-radius-m 0.5 \
  --command-timeout-s 1.0 \
  --anchor-id car_01_anchor \
  --anchor-x-m 10 \
  --anchor-y-m 10 \
  --anchor-yaw-rad 0

# 独立 Pygame 运动/碰撞调试窗口
mockvehicle2d visual

# 完整真值地图上的 A* 离线调试工具，不属于自主运行链路
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200
```

依赖安装在仓库本地 `.venv/`。公开接口统一使用 SI 单位：米、秒、弧度、米/秒和
弧度/秒。

## 控制方式

协议版本为 `4`，不接受旧的 `cmd`、`drive`、直接 `goto` 或自然语言命令。
每个连接内的 `seq` 必须严格递增。

```json
{"type":"mode","seq":1,"action":"switch_to_manual"}
{"type":"manual","seq":2,"action":"drive","linear_mps":0.3,"angular_rps":-0.4}
{"type":"manual","seq":3,"action":"stop"}
{"type":"mode","seq":4,"action":"stop_motion"}

{"type":"mode","seq":5,"action":"switch_to_auto"}
{"type":"auto","seq":6,"action":"push","missions":[
  {"mission_id":"goto-001","type":"goto","frame_id":"global_map","x_m":20.0,"y_m":30.0}
]}
{"type":"auto","seq":7,"action":"pause"}
{"type":"auto","seq":8,"action":"resume"}
{"type":"auto","seq":9,"action":"cancel_all"}
```

模式语义：

- 模式切换先停车；重复切到当前模式是无副作用的幂等操作。
- `mode/stop_motion` 在 Manual、Auto 或模式切换竞态中都立即停车；Auto 任务暂停并
  保留，重复调用不产生重复事件。
- 手动命令只在 `manual` 模式有效，自动命令只在 `auto` 模式有效。
- 手动 `drive` 是有租约的连续速度设定值；客户端需在
  `command-timeout-s` 内持续刷新，松手发送 `stop`。
- Auto → Manual 会暂停当前任务并保留队列；切回 Auto 后仍为 `paused`，需要显式
  `resume`。
- `pause` 保留活动任务和队列；`cancel_all` 才会清空它们。
- Auto 已在执行时重复 `resume` 是无副作用操作，不会停车或重启规划。
- `mission_id` 是幂等键。相同 ID 和相同目标的重试不会重复入队；相同 ID 携带不同目标
  会被拒绝。
- 控制连接断开时车辆立即停车，自动任务暂停而不是丢弃；重连后可显式恢复。
- 非法输入触发故障停车；活动自动任务进入暂停状态。

每条合法命令先收到 `command_ack`。自动任务另外通过 `mission_update` 报告
`queued`、`active`、`paused`、`reached`、`blocked` 和 `cancelled`。
完整字段见 [WebSocket 协议](docs/websocket_protocol.md)。

## 自主导航

每辆车启动时只知道 `global_map → anchor_map` 的出生锚点。之后由运动增量生成锚定
odometry，并将 Tmini 扫描累计到车辆自己的
`ObservedGrid`（Unknown / Free / Occupied / Forbidden）。

自动任务只读取该局部位姿和有限视野地图：

- Unknown 可通行但代价较高，驶向 Unknown 时降速；
- 新障碍进入 Tmini 视野并改变局部地图后，D* Lite 增量修补路线；
- 每个控制帧限制 D* 扩展和安全候选检查数量，规划未完成时保持停车；
- 精确目标不可安全到达时，在车体外缘距目标不超过 `1 m` 的范围内选择已确认安全位置；
- 定位 `degraded` 时自动降速，`lost` 时任务阻断；
- 水平 Tmini 不能发现落差，模拟器使用独立的下视安全输入。

完整说明见 [有限视野寻路](docs/pathfinding.md)。

## 主要文件

```text
src/mockvehicle2d/
├── controller.py          # 模式、队列、任务生命周期和唯一控制权
├── protocol.py            # WebSocket v4 严格 JSON 边界
├── server.py              # 独占连接、帧调度和遥测
├── navigation.py          # 有限视野 Goto 决策
├── local_state.py         # 锚点、odometry、scan matching、ObservedGrid
├── safety.py              # 障碍/边缘净空和故障停车
├── scan.py                # YDLidar Tmini 二维扫描
├── vehicle.py             # 运动学、看门狗和执行器状态
├── collision.py           # 连续碰撞检查
├── map_grid.py            # 仿真真值环境
└── pathfinding/
    ├── d_star_lite.py     # 在线增量规划
    └── a_star.py          # 离线全真值调试
```

## 仿真边界

`map_full.source=simulator_ground_truth` 只用于 Pictor 调试显示、物理碰撞和生成传感器
读数。自主控制器不会读取它。当前定位仅包含增量 odometry 和有限窗 scan matching，
没有回环、位姿图、全局优化、地图持久化或中央地图同步；因此它是控制架构与算法测试台，
不是现实传感器和车辆动力学的完整数字孪生。

可选的自然语言意图翻译器只生成合法 v4 命令，不接入 WebSocket 执行环，也不持有
控制权。接口和限制见
[自然语言边界文档](docs/nl_function_calling_design.md)。
