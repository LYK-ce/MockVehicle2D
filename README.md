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
  --realtime-factor 5 \
  --mission-capacity 16 \
  --linear-speed-mps 0.5 \
  --angular-speed-rps 1.5708 \
  --linear-acceleration-mps2 1.0 \
  --linear-deceleration-mps2 1.0 \
  --angular-acceleration-rps2 3.1416 \
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

# 不启动 Godot、WebSocket 或墙钟等待，按固定模拟 tick 执行一次实验
mockvehicle2d episode \
  --scenario examples/single_vehicle_episode.json \
  --max-simulation-s 30 \
  --goto mock_vehicle_01,11,10

# 无 localhost libp2p 的双车交叉基准；进程内 peer-state + motion-intent relay 驱动确定性让行
mockvehicle2d episode \
  --scenario examples/two_vehicle_crossing_episode.json \
  --max-simulation-s 30 \
  --goto mock_vehicle_01,11,11 \
  --goto mock_vehicle_02,9,11

# 四车从东、西、南、北穿过同一中心的空场确定性协同基准
python -m pytest -p no:cacheprovider -q tests/test_episode.py \
  -k four_vehicle_crossing
```

依赖安装在仓库本地 `.venv/`。公开接口统一使用 SI 单位：米、秒、弧度、米/秒、
弧度/秒、米/秒²和弧度/秒²。车辆区分控制器请求的 target velocity 与实际 executed
velocity；默认线加速、线减速和角加速上限分别为 `1 m/s²`、`1 m/s²` 和 `π rad/s²`。
普通 stop、watchdog 和 safety stop 将 target 置零并按上限制动，正反向切换先减到零；
静态碰撞和多车同时仲裁拒绝候选轨迹时才立即钳制实际速度，以保持不穿透。

`serve` 和 `fleet` 默认以 `--realtime-factor 5` 运行，即固定物理步长、传感器
周期、控制阈值和 P2P 模拟时序不变，只把墙钟等待缩短为原来的五分之一；传入 `1`
可恢复原来的实时速度。

## Headless Episode Runner

`episode` 复用 `FleetRuntime`、`RobotController` 和现有 Mission 语义，在调用线程中直接
推进固定模拟 tick。它不读取墙钟、不打开 WebSocket，也不受 `serve/fleet` 的
`realtime_factor` 影响。CLI 至少需要一个 `--goto VEHICLE_ID,X_M,Y_M`；可重复该参数为
一辆或多辆车依次入队。任务达到全部完成或任一阻断后，Runner 继续按固定 tick 记录
有界制动尾段，直到相关车辆的 target 和 executed velocity 都归零；制动仍受
`--max-simulation-s` 限制。达到全部已提交任务并完成制动后成功结束，任务阻断或达到
时限时失败结束。`episode` 与 `serve`/`fleet` 使用相同的速度、加减速、半径和 watchdog
CLI 参数。

多车 Episode 不启动 localhost libp2p，而是把现有 peer-state v1 和 motion-intent v3
payload 经过 JSON 序列化及协议校验后，以固定 1 tick 延迟在进程内传递；序列、接收时间
和 `0.35 s` 过期规则与实时 P2P 路径一致。启用了真实 `p2p` 配置的场景仍被拒绝，因为
libp2p 墙钟调度和 map delta 传播不属于确定性 Episode。每辆车沿 OwnMap D* 路径发布
约 `4 s` 的相对 cell 时间窗；接收端以 receipt time 重建 cell/edge/goal reservation，加入
车体、安全余量、加减速和通信不确定性后运行 prioritized SIPP，每次只执行前 `0.8 s`
并滚动重算。同格、对向 edge swap、同步几何交叉和已到达车辆的 goal hold 都进入冲突
检查；无冲突路径不增加等待。

未提交候选先比较等待年龄，再按继承 owner 和 `vehicle_id` 排序；一旦进入短 commit，
仲裁只使用所有节点都能看到的 owner/vehicle ID，不允许未同步的本地年龄抢占。协议中的
活动任务年龄当前仅供观测，跨车“任务更早”需要后续引入 Lamport task token 才能安全参与
全序。占路车辆会沿 request chain 递归继承优先级，并只尝试一个本地可通行邻格作为有界
回溯。第一版没有独立的网络 propose/ACK/commit 往返；commit 依赖上一 tick 的全员 fresh
intent、确定性优先级和短执行前缀，并继续由下一格租约、物理碰撞仲裁与 LocalSafety
兜底。peer state 或 intent 缺失/过期、sidecar 未 ready 或 generation 回退时保持停车。

完全观测且内宽不超过约 `3 m` 的直线瓶颈还会使用 motion-intent v3 的有向 corridor
descriptor 做短租约仲裁。只有一个经对端确认的 owner 可以进入；与 owner 方向相反的
失败者按距出口侧 entry 的纵向距离和 `vehicle_id` 选出唯一 front waiter，提前进入可逆
侧向等待，同侧 rear waiter 原地排队。ACK 只确认 owner，不等于入口已安全；
front waiter 的连续位姿和剩余侧移段未离开 `2r + 0.3 m` 包络前，owner 只能接近入口并在
物理制动距离外停车。释放要求整个车体和 `0.3 m` 自动安全余量越过远端边界。这是 PIBT
启发的局部租约机制，不是完整 PIBT、MAPF 或中央车队调度器。

新 owner 只有在 P2P ready 且本 session 的每个 expected vehicle 都有未过期 motion intent
时才能确认；`corridor:null` 也算明确的无竞争声明，因此独立走廊不会被全局串行。peer
断联或 intent 过期时，入口外的新 claim fail-closed；已经确认或已入廊的 owner 保留租约
直到清空并释放。未启用 P2P 且没有已知 peer 的真正单车仍按连续 3 tick 自确认。

标准输出是 schema version 2 的单行 canonical JSON，包含场景 ID、odometry seed、tick 数、
模拟时长、终止原因，以及每辆车的仿真真值终态、按 tick 采样的路径长度、碰撞/阻断/
安全终态和任务状态。顶层 `minimum_inter_vehicle_clearance_m` 从 `t=0` 开始，取所有固定
tick、所有无序车辆对的最小圆形 footprint 边缘间距（中心距减去两车半径）；单车时为
`null`，负值表示 footprint 已重叠。每车 `longest_no_progress_duration_s` 是连续固定 tick
内真值中心每 tick 平移小于 1 mm 的最长模拟时长；只在该车仍有未完成任务时统计，原地
转向和等待计入，恢复至少 1 mm 平移后重新计数。某车完成全部任务后等待其他车辆的正常
停驻以及 episode 终止后的制动不计入。真值只由评估层读取，不会进入自主控制链。Python
调用入口为
`mockvehicle2d.episode.run_episode`，可直接传入现有 `GotoMission`、`PatrolMission` 或
`CoverageMission`。

Runner 明确拒绝启用真实 P2P 的场景，因为 localhost libp2p 调度和 map delta 传播不属于
确定性模拟时钟；当前只提供上述进程内 peer-state + motion-intent relay，不包含定时事件或通信故障注入。

四车任务协同回归使用 `tests/fixtures/four_vehicle_mission_matrix.json` 和显式空场，覆盖
1/2/4 车互不冲突 Goto、四车循环换位、相邻 Goto 终点、终点车辆挡路绕行、重复共享
路口 Patrol、对向合流 Patrol、互不相交 Patrol、相邻静态条带 Coverage、共享入口四象限
Coverage，以及 Goto/Patrol/Coverage 混合任务。每项任务仍归接收命令的原车，协调只影响
执行期运动。Coverage 分区由测试输入静态分配，接缝小于一个 `0.5 m` 本地网格；它不引入
中央地图，也不把 peer evidence 写入 OwnMap。

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_episode.py \
  -k 'disjoint_goto or parked_goal_vehicle or four_vehicle and (cycle or adjacent or patrol or coverage or mixed)'
```

耗时更长的扩展矩阵选择最困难的对向合流 Patrol 和共享入口 Coverage，分别验证同 seed
重复、车辆声明反序以及 `50/250 ms` tick。当前开发机约需 17 分钟，因此默认 deselected：

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q -m extended tests/test_episode.py -k extended_matrix
```

四车争用同一条 `5 m` 单车道的严格串行验收也标为 extended。当前策略不组成同向 convoy；
验收给进场、可逆侧移/回归、前车出口冲突和末段共 `130 s` 确定性完成预算，并把深队列
的连续活动等待限制在 `90 s`：

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q -m extended \
  tests/test_motion_coordination.py \
  -k four_vehicles_share_one_corridor
```

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

每车严格维护 OwnMap、按来源划分的 PeerEvidence 和只读 CollaborativeView。远端地图证据
不会进入 OwnMap、不会重新发布，也暂不改变 D* Lite 或本地安全决策；peer vehicle state
只作为瞬态 footprint exclusion 和让行输入；motion intent 只进入租约仲裁，不进入
OwnMap、PeerEvidence 或 CollaborativeView。P2P 健康度、
本地/远端 delta 计数和协同视图摘要通过每车 `pose.p2p_map_sync` 遥测提供。当前仅实现
在线增量和重复/过期拒绝；离线缺包与后加入节点的 tile 快照恢复留到下一里程碑。

Gossipsub topic 为 `mockvehicle2d/<session_id>/fleet-sync/1`，payload 是严格的
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

同一 fleet-sync topic 还承载严格的 `mockvehicle2d-motion-intent/3` JSON。它只描述由
本车 odometry 与 OwnMap D* 路径生成的短时域计划，不携带仿真真值：

```json
{
  "protocol": "mockvehicle2d-motion-intent/3",
  "session_id": "four_vehicle_exploration",
  "source_vehicle_id": "mock_vehicle_01",
  "intent_generation": 1,
  "sequence": 9,
  "frame_id": "global_map",
  "resolution_m": 1.0,
  "timestamp_s": 42.1,
  "lease_duration_s": 0.35,
  "plan_generation": 3,
  "current_cell": {"gx": 9, "gy": 10},
  "target_cell": {"gx": 10, "gy": 10},
  "priority": {
    "wait_ticks": 3,
    "task_age_ticks": 27,
    "task_sequence": 6,
    "owner_vehicle_id": "mock_vehicle_01"
  },
  "reserved": true,
  "trajectory": [
    {"cell":{"gx":9,"gy":10},"enter_offset_s":0.0,"leave_offset_s":0.2},
    {"cell":{"gx":10,"gy":10},"enter_offset_s":0.7,"leave_offset_s":1.0}
  ],
  "committed_until_offset_s": 0.8,
  "goal_hold": false,
  "safety_time_margin_s": 0.6,
  "corridor": {
    "entry_cell": {"gx": 9, "gy": 10},
    "exit_cell": {"gx": 14, "gy": 10}
  }
}
```

接收端严格校验 intent/plan generation、sequence、global frame、resolution、cell 边界、
最多 64 个且 `0..4 s` 单调的相对时间窗、commit/margin 上限、优先级 owner 白名单、租约
上限和有向 axis-aligned corridor；重复、乱序、超时或额外字段均拒绝。相对时间以接收时刻
重建，不要求不同车辆的 monotonic clock 同步；sender timestamp 只用于同源防回退。
没有走廊声明时 `corridor` 必须为 `null`。走廊只从车辆 OwnMap 上已完全观测的直线窄通道
推导；peer evidence 和仿真真值不会参与检测。相反方向看到的部分 descriptor 会按重叠轴
匹配并单调扩展释放边界，进入前需要一轮 owner 声明/对端确认。

当前走廊策略严格一次只放行一辆车，因此长走廊和深队列的吞吐近似线性增长。暂未实现
“暂不可入廊 owner”跳过或同向 directional batching/convoy。弯曲、分支、过宽或观测
不完整的通道使用通用 SIPP 时间窗，但第一版只在一条 D* 空间候选及一个邻格 detour 上
做调度，不是联合最优 MAPF，也不宣称完整 PIBT 回溯。

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
  {"mission_id":"goto-001","type":"goto","frame_id":"global_map","x_m":20.0,"y_m":30.0},
  {"mission_id":"patrol-001","type":"patrol","frame_id":"global_map","waypoints":[{"x_m":20.0,"y_m":30.0},{"x_m":24.0,"y_m":30.0}],"cycles":2},
  {"mission_id":"coverage-001","type":"coverage","frame_id":"global_map","area":{"min_x_m":20.0,"min_y_m":20.0,"max_x_m":30.0,"max_y_m":25.0},"lane_spacing_m":1.0}
]}
{"type":"auto","seq":7,"action":"pause"}
{"type":"auto","seq":8,"action":"resume"}
{"type":"auto","seq":9,"action":"cancel_all"}
```

模式语义：

- 模式切换先请求制动；重复切到当前模式是无副作用的幂等操作。
- `mode/stop_motion` 在 Manual、Auto 或模式切换竞态中都请求有界制动；Auto 任务暂停并
  保留，重复调用不产生重复事件。
- 手动命令只在 `manual` 模式有效，自动命令只在 `auto` 模式有效。
- 手动 `drive` 是有租约的连续速度设定值；客户端需在
  `command-timeout-s` 内持续刷新，松手发送 `stop`。
- Auto → Manual 会暂停当前任务并保留队列；切回 Auto 后仍为 `paused`，需要显式
  `resume`。
- `pause` 保留活动任务和队列；没有活动或排队任务时保持 `Idle`；`cancel_all`
  才会清空任务。
- `patrol` 按给定航点执行有限轮次；`coverage` 从矩形左下角开始，沿长边生成蛇形
  路线。二者都复用现有 `goto` 导航，每个父任务最多生成 1024 个子目标且只占一个
  队列位置。
- Auto 已在执行时重复 `resume` 是无副作用操作，不会停车或重启规划。
- `mission_id` 在 Server 进程生命周期内是永久幂等键。相同 ID 和完全相同任务定义的
  重试不会重复入队；相同 ID 携带不同定义会被拒绝。进程重启后该内存状态会清空。
- 控制连接断开时车辆请求有界制动，自动任务暂停而不是丢弃；重连后可显式恢复。
- 非法输入触发故障停车；活动自动任务进入暂停状态。

每条合法命令先收到 `command_ack`。自动任务另外通过 `mission_update` 报告
`queued`、`active`、`paused`、`reached`、`blocked` 和 `cancelled`。
高层任务的事件和控制器快照携带当前子目标及 0-based `subgoal_index/subgoal_count`；
中间子目标不会产生独立任务或 `reached`，父任务只在最后一个子目标完成时到达。
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
├── episode.py             # 固定 tick 的 headless 实验执行与结果
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
