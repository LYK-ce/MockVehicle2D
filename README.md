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

# 完整真值地图上的 A* 离线调试工具，不属于自主运行链路
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200

# 默认运行两车（19090～19091）；四车测试改用 examples/four_vehicle_scenario.json
cargo build --bin map-sync-node
mockvehicle2d fleet --scenario examples/two_vehicle_scenario.json
```

依赖安装在仓库本地 `.venv/`。公开接口统一使用 SI 单位：米、秒、弧度、米/秒和
弧度/秒。

## 多车共享世界

`fleet` 入口在一个进程中运行唯一的确定性物理世界和 1～4 个隔离车辆节点。每个节点
拥有自己的 `RobotController`、任务队列、odometry、`ObservedGrid`、D* Lite、控制租约
和 WebSocket endpoint；节点不能读取其他车辆的控制状态、地图或世界真值。

```text
SharedWorld（仅仿真器可见）
  ├── 静态真值、统一 100 ms tick、Tmini 射线和碰撞仲裁
  ├── RobotNode 01 ── ws://127.0.0.1:19090
  ├── RobotNode 02 ── ws://127.0.0.1:19091
  ├── RobotNode 03 ── ws://127.0.0.1:19092
  └── RobotNode 04 ── ws://127.0.0.1:19093
```

场景 JSON 为每辆车声明唯一的 `vehicle_id`、`operator_port`、`spawn_id` 和
`anchor_pose {x_m,y_m,yaw_rad}`。启动是原子的：车辆数量、身份、端口、出生点、世界
边界、静态障碍、安全余量和车间重叠全部通过后才创建任何车辆。真值初始位姿等于
`anchor_pose`，但车辆自己的 odometry 从 `(0,0,0)` 开始，只通过运动增量和传感器更新。

每个 tick 对所有车辆使用同一快照，先独立控制，再计算候选运动、统一处理静态/车间
碰撞并同时提交。其他车辆会产生动态 Tmini 回波和物理碰撞，但动态回波不会永久写成
本车 `ObservedGrid` 的静态 Occupied，避免车辆离开后留下幽灵障碍。

Pictor 中分别创建以上四个 WebSocket 连接即可同时显示四辆车。

示例场景还为四车启动四个独立的 Rust `map-sync-node` 进程，使用 TCP
`20090～20093` 组成固定 localhost libp2p Gossipsub mesh。`cargo test` 会构建并验证
sidecar；默认无顶层 `p2p` 的场景不启动任何网络进程。每辆车每 100 ms 最多发布一批
自己 `ObservedGrid` 新产生的 dirty cells；没有变化时不发消息，发送和接收都不会等待
其他节点。身份密钥保存在配置的 `runtime_dir`，进程重启后 PeerId 保持稳定。

每车严格维护 OwnMap、按来源划分的 PeerEvidence 和只读 CollaborativeView。远端证据
不会进入 OwnMap、不会重新发布，也暂不改变 D* Lite 或本地安全决策。P2P 健康度、
本地/远端 delta 计数和协同视图摘要通过每车 `pose.p2p_map_sync` 遥测提供。当前仅实现
在线增量和重复/过期拒绝；离线缺包与后加入节点的 tile 快照恢复留到下一里程碑。

Gossipsub topic 为 `mockvehicle2d/<session_id>/map-delta/1`，payload 是严格的
`mockvehicle2d-map-delta/1` JSON：

```json
{
  "protocol": "mockvehicle2d-map-delta/1",
  "session_id": "four_vehicle_exploration",
  "source_vehicle_id": "mock_vehicle_01",
  "map_epoch": 1,
  "sequence": 7,
  "source_frame": "anchor_map",
  "anchor_id": "spawn_north_west",
  "transform_epoch": 1,
  "transform_to_global_map": {"x_m": 9.0, "y_m": 9.0, "yaw_rad": 0.0},
  "resolution_m": 1.0,
  "cells": [{"gx": 3, "gy": 4, "state": 1}]
}
```

接收端同时校验 Gossipsub 签名作者 PeerId、车辆白名单、session、frame、anchor、epoch、
resolution、sequence、消息大小和每个 cell。Python 与对应 sidecar 只通过运行目录中的
Unix domain socket 交换有界 JSONL 消息，控制 tick 不等待该 socket。

## 控制方式

协议版本为 `4`，只接受 `mode`、`manual` 和 `auto` 三类命令。
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
- `pause` 保留活动任务和队列；没有活动或排队任务时保持 `Idle`；`cancel_all`
  才会清空任务。
- Auto 已在执行时重复 `resume` 是无副作用操作，不会停车或重启规划。
- `mission_id` 在 Server 进程生命周期内是永久幂等键。相同 ID 和相同目标的重试不会
  重复入队；相同 ID 携带不同目标会被拒绝。进程重启后该内存状态会清空。
- 控制连接断开时车辆立即停车，自动任务暂停而不是丢弃；重连后可显式恢复。
- 非法输入触发故障停车；活动自动任务进入暂停状态。

每条合法命令先收到 `command_ack`。自动任务另外通过 `mission_update` 报告
`queued`、`active`、`paused`、`reached`、`blocked` 和 `cancelled`。
每个任务事件携带 `event_epoch` 和进程内严格递增的 `event_seq`。Server 保留本进程
产生的全部任务事件，连接建立后按顺序自动重放；断线重连可能再次收到已经处理过的
事件，客户端应按 `(event_epoch, event_seq)` 幂等去重。命令 ack 只确认命令是否受理，
任务结果以事件为准。
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
├── fleet.py               # 1～4 车场景、共享物理世界和独立 endpoint
├── map_sync.py            # OwnMap 增量、PeerEvidence、UDS sidecar 生命周期
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

Rust sidecar 位于 `rust/map_sync_node.rs`：每车使用独立签名 PeerId、TCP+Noise+Yamux 和
Gossipsub；固定四车仿真只配置 localhost peer，不启用 DHT、relay、NAT 或公网发现。

## 仿真边界

`map_full.source=simulator_ground_truth` 只用于 Pictor 调试显示、物理碰撞和生成传感器
读数。自主控制器不会读取它。当前定位仅包含增量 odometry 和有限窗 scan matching，
没有回环、位姿图、全局优化、地图持久化或中央地图同步；因此它是控制架构与算法测试台，
不是现实传感器和车辆动力学的完整数字孪生。
