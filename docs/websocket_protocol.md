# WebSocket v4 协议

## 基本约定

| 项目 | 值 |
|------|----|
| 角色 | 小车是 WebSocket Server，控制端是 Client |
| 默认地址 | `ws://127.0.0.1:19090` |
| 命令格式 | UTF-8 JSON 文本对象 |
| 协议版本 | `4` |
| 控制权 | 每辆车同时只有一个独占连接 |
| 单位 | m、s、rad、m/s、rad/s、m/s²、rad/s² |
| 任务坐标系 | `global_map` |
| 默认时间倍率 | `5.0`（`--realtime-factor 1` 恢复实时） |

多车模式保持同一个 v4 协议，每辆车使用独立 endpoint。示例四车场景监听
`19090`～`19093`；一个 endpoint 的连接、命令序号和独占租约不会影响其他车辆。

下行命令的顶层字段必须与定义完全一致，不允许缺失或额外字段。JSON 重复 key、
`NaN`、`Infinity`、二进制命令、超过 64 KiB 的消息均无效。每个连接内的 `seq` 是
无符号 64 位整数且必须严格递增。

除 `mode`、`manual`、`auto` 外的消息类型，以及无 `type/seq` 的格式都不属于 v4。

## 连接流程

```text
Client ── WebSocket handshake ──► Vehicle
Client ◄──────── hello ────────── Vehicle
Client ◄── retained mission_update ─ Vehicle  （按 event_seq 重放）
Client ◄──── binary map chunks ── Vehicle
Client ◄──── pose + scan ──────── Vehicle  （约 6 Hz）
Client ─────── command ─────────► Vehicle
Client ◄──── command_ack ──────── Vehicle
Client ◄──── mission_update ───── Vehicle  （有状态变化时）
```

已有控制连接时，新连接只收到：

```json
{
  "type": "error",
  "timestamp_s": 1717800000.1,
  "seq": null,
  "code": "vehicle_busy",
  "message": "another controller owns the vehicle lease"
}
```

断开会立即请求有界制动并释放租约。odometry、本地地图、活动任务和待执行队列保留；
有未完成任务时自动状态变为 `paused`。重连后由 `hello.controller` 恢复当前状态，
随后 Server 按 `event_seq` 重放本进程保留的全部任务事件，再显式 `resume`。

## 下行命令

### 模式

```json
{"type":"mode","seq":1,"action":"switch_to_auto"}
{"type":"mode","seq":2,"action":"switch_to_manual"}
{"type":"mode","seq":3,"action":"stop_motion"}
```

模式命令不受当前模式限制。实际切换先请求有界制动；重复切到当前模式是幂等操作。
Auto → Manual 暂停并保留任务。Manual → Auto 若有保留任务仍停在 `paused`，需要
`auto/resume`。

`stop_motion` 是模式无关的安全停车入口：

- 在 Manual 中立即请求有界制动并清除手动速度租约，模式仍为 Manual；
- 在 Auto 中立即请求有界制动，活动任务和队列保留，状态变为 `paused`；
- 在 Auto Idle 中保持 Idle；
- 重复调用不重复发布 paused 事件。

### 手动速度

```json
{
  "type": "manual",
  "seq": 4,
  "action": "drive",
  "linear_mps": 0.25,
  "angular_rps": -0.4
}
```

```json
{"type":"manual","seq":5,"action":"stop"}
```

仅在 `manual` 模式有效。线速度与角速度必须是有限 JSON 数字，绝对值不得超过 Server
的 `--linear-speed-mps` 和 `--angular-speed-rps`。

`drive` 是有时限的设定值：客户端在按住操作键时持续刷新，刷新间隔必须小于
`--command-timeout-s`；松手发送 `stop`。设定值经过本地安全门控后才安装到车辆。

### 自动任务入队

```json
{
  "type": "auto",
  "seq": 5,
  "action": "push",
  "missions": [
    {
      "mission_id": "goto-001",
      "type": "goto",
      "frame_id": "global_map",
      "x_m": 20.0,
      "y_m": 30.0
    },
    {
      "mission_id": "patrol-001",
      "type": "patrol",
      "frame_id": "global_map",
      "waypoints": [
        {"x_m": 24.0, "y_m": 35.0},
        {"x_m": 28.0, "y_m": 35.0}
      ],
      "cycles": 2,
      "coordination_id": "fleet-patrol-001"
    },
    {
      "mission_id": "coverage-001",
      "type": "coverage",
      "frame_id": "global_map",
      "area": {
        "min_x_m": 10.0,
        "min_y_m": 10.0,
        "max_x_m": 20.0,
        "max_y_m": 15.0
      },
      "lane_spacing_m": 1.0,
      "coordination_id": "fleet-coverage-001"
    }
  ]
}
```

仅在 `auto` 模式有效。支持：

- `goto`：单个 `x_m/y_m` 目标；
- `patrol`：按 `waypoints` 顺序执行正整数次 `cycles`；可选 `coordination_id` 使用与
  `mission_id` 相同的 1–64 字符格式，省略时保持原路线；
- `coverage`：覆盖有效矩形 `area`。从 `(min_x_m,min_y_m)` 开始，沿矩形长边往返，
  相邻横道间距为正数 `lane_spacing_m`；短边不能整除间距时仍包含末端边界。可选
  `coordination_id` 使用与 `mission_id` 相同的 1–64 字符格式；省略时每辆车执行完整区域。

设置 Patrol/Coverage 的 `coordination_id` 时，任务首次激活会冻结按 `vehicle_id` 排序的
`{本车} + expected peer allowlist`。Patrol 的起始索引为
`floor(member_rank * waypoint_count / member_count)`，循环旋转航点后仍完整执行全部航点和
`cycles`；成员多于航点时允许确定性重复起点。Coverage 沿原区域长轴获得连续子矩形，
继续使用相同蛇形路线和 Cooperative Goto。同组全部固定成员必须收到相同的
`coordination_id` 和任务参数；当前不支持只向部分 expected peer 下发、动态加入/退出或
运行中重分配。该字段不创建新的车队任务类型，也不更改 P2P 或 motion-intent v4 schema。

三类任务都使用 `global_map`。`mission_id` 为 1–64 个 ASCII 字母、数字、点、
下划线、冒号或连字符。所有坐标必须有限且绝对值不超过 `1,000,000 m`。
`patrol` 航点不能为空，`coverage` 的每个 minimum 必须小于对应 maximum。一个父任务
最多生成 1024 个子目标；在生成路线前检查该上限，因此极小间距不会分配无界内存。
高层任务通过现有 `goto` 导航逐个执行子目标，不创建独立子任务。

一次 push 是原子的：

- batch 不能为空、不能超过任务队列配置，且 batch 内 ID 必须唯一；
- 待执行队列空间不足时整个 batch 拒绝，高层任务按一个父任务计数；
- 任一任务或生成路线无效时整个命令在进入控制器前拒绝，不会部分入队；
- 已知 `mission_id` 携带完全相同的类型和任务定义时视为幂等重试，不重复入队；
- 已知 `mission_id` 携带不同类型或定义时返回 `mission_id_conflict`。

客户端重试时必须使用新的、更大的命令 `seq`，但沿用原 `mission_id`。
幂等记录在 Server 进程生命周期内不会静默淘汰；进程重启后内存记录清空，不提供
跨重启幂等或持久化恢复。

### 自动任务控制

```json
{"type":"auto","seq":6,"action":"pause"}
{"type":"auto","seq":7,"action":"resume"}
{"type":"auto","seq":8,"action":"cancel_all"}
```

| action | 语义 |
|--------|------|
| `pause` | 停车，保留活动任务和队列 |
| `resume` | 从 Paused/Blocked 重新启动；已为 Active 时幂等，无任务时进入 idle |
| `cancel_all` | 停车，取消活动任务并清空队列 |

任务 `blocked` 后不会自动跳到下一项。控制端可以修复条件后 `resume` 重试，或
`cancel_all` 放弃整批任务。

没有活动任务和待执行任务时，`pause` 是无副作用操作，状态保持 `Auto/Idle`。

## 命令确认

每个通过语法边界的命令返回一个 `command_ack`：

```json
{
  "type": "command_ack",
  "timestamp_s": 1717800000.2,
  "seq": 5,
  "command": {"type": "auto", "action": "push"},
  "accepted": true,
  "controller": {
    "mode": "auto",
    "auto_state": "active",
    "active_mission": null,
    "mission_queue": {
      "size": 2,
      "capacity": 16,
      "mission_ids": ["goto-001", "goto-002"]
    },
    "manual_setpoint_active": false,
    "navigation": {"status": "idle"}
  }
}
```

`accepted=true` 表示命令已被控制器接受，不表示任务到达，也不保证首个规划切片已完成。
拒绝时增加 `reason`，例如：

| reason | 含义 |
|--------|------|
| `wrong_mode` | 命令族与当前模式不符 |
| `mission_queue_full` | 待执行队列空间不足 |
| `mission_id_conflict` | 相同任务 ID 对应不同内容 |
| `safety_obstacle` / `safety_edge` | 手动速度被安全门控拒绝 |
| `safety_sensor_fault` | 安全输入故障 |

## 任务事件

任务状态变化通过独立的 `mission_update` 上报：

```json
{
  "type": "mission_update",
  "event_epoch": "c1df10b7f5cd48a5a4850665b16bb1f8",
  "event_seq": 42,
  "timestamp_s": 1717800000.3,
  "mission_id": "goto-001",
  "mission_type": "goto",
  "submitted_seq": 5,
  "status": "blocked",
  "subgoal_index": 0,
  "subgoal_count": 1,
  "goal": {
    "frame_id": "global_map",
    "x_m": 20.0,
    "y_m": 30.0
  },
  "reason": "no_path",
  "detail": "nearby_safe_goal_unavailable",
  "navigation": {
    "status": "blocked",
    "goal_mode": "approaching_safe_stop",
    "planning": false
  }
}
```

`event_epoch` 标识当前内存 ledger，Server 进程重启后变化；`event_seq` 从 1 开始，
在该 epoch 内严格递增且不复用。事件先写入
`RobotController` 的进程内 ledger，再尝试发送；发送失败不会删除事件。一个连接内只按
递增顺序发送尚未发送的事件。重连会从 ledger 起点自动重放，因此传输语义是
**at-least-once**：跨连接可能重复，但不会改变事件身份或乱序。客户端必须按
`(event_epoch, event_seq)` 去重；epoch 变化时清除旧的 sequence 游标。

`subgoal_index` 从 0 开始；`goal` 是该事件对应的当前子目标。中间目标不会使用新的
`mission_id`，也不会单独发布任务事件；连续进度由 `pose.controller.active_mission`
提供。只有最后一个子目标完成才发布父任务 `reached`；`active`、`paused`、`blocked`
和 `cancelled` 保留事件发生时的子目标进度。

当前 simulator 为便于验证可靠性，保留本进程的全部任务事件；进程重启后 ledger 和
序号都会清空。`timestamp_s` 是本次发送对应的模拟时间，重放时可能变化，事件身份只能使用
`(event_epoch, event_seq)`。

状态及触发：

| status | 含义 |
|--------|------|
| `queued` | 新任务已进入队列 |
| `active` | 任务已成为活动项并开始规划 |
| `paused` | `stop_motion`、模式接管、显式暂停、非法输入或连接断开 |
| `reached` | 精确目标或附近安全目标已到达 |
| `blocked` | 无路、定位丢失、安全故障或目标超出规划硬限制 |
| `cancelled` | `cancel_all` 取消 |

规划距离超过硬限制时，任务先正常收到 push 的 ack/queued，随后收到：

```json
{
  "type": "mission_update",
  "mission_id": "too-far",
  "status": "blocked",
  "reason": "invalid_goal",
  "detail": "goal exceeds maximum distance"
}
```

该任务错误不会中断 WebSocket。

## 上行遥测

### hello

`hello` 是业务首帧：

```json
{
  "type": "hello",
  "protocol_version": 4,
  "vehicle_id": "mock_vehicle_01",
  "control_lease": "exclusive",
  "mission_frame_id": "global_map",
  "mission_types": ["goto", "patrol", "coverage"],
  "realtime_factor": 5.0,
  "birth_anchor": {
    "anchor_id": "spawn_north_west",
    "x_m": 9.0,
    "y_m": 9.0,
    "yaw_rad": 0.0
  },
  "map": {
    "source": "simulator_ground_truth",
    "frame_id": "simulator_map",
    "resolution_m": 1.0,
    "width_cells": 256,
    "height_cells": 256,
    "transform_to_global_map": {
      "x_m": 0.0,
      "y_m": 0.0,
      "yaw_rad": 0.0
    },
    "binary_chunks": {
      "type": 0,
      "chunk_size_cells": 256,
      "header": ">Bii",
      "byte_order": "big",
      "payload_order": "row_major_y_x"
    }
  },
  "controller": {
    "mode": "manual",
    "auto_state": "idle",
    "mission_events": {
      "event_epoch": "c1df10b7f5cd48a5a4850665b16bb1f8",
      "latest_event_seq": 0,
      "retention": "process_lifetime"
    }
  }
}
```

`realtime_factor` 表示一秒墙钟时间内推进的模拟秒数。倍率只改变墙钟节拍；所有消息的
`timestamp_s` 共用同一模拟 epoch，物理量、控制/传感器周期、命令超时与 P2P 100 ms
发布周期仍按模拟时间计算。`map.source=simulator_ground_truth` 表示调试真值，不能作为
自动规划输入。

### 二进制地图 chunk

`hello` 后发送一个或多个二进制帧：

```text
offset  bytes  encoding
0       1      u8 frame_type = 0
1       4      i32 big-endian chunk_origin_gx
5       4      i32 big-endian chunk_origin_gy
9       65536  256×256 u8 cells，row-major y/x
```

cell 状态：`0` 可通行、`1` 墙、`2` 无地面/落差。客户端用
`hello.map.transform_to_global_map` 将 `simulator_map` 坐标叠加到 `global_map`。

### pose

```json
{
  "type": "pose",
  "timestamp_s": 1717800000.4,
  "seq": 12,
  "source": "anchored_odometry",
  "frame_id": "global_map",
  "x_m": 12.1,
  "y_m": 8.5,
  "z_m": 0.0,
  "yaw_rad": 0.4,
  "vx_mps": 0.2,
  "vy_mps": 0.08,
  "omega_rps": 0.1,
  "collision": false,
  "actuator_command": "drive",
  "controller": {
    "mode": "auto",
    "auto_state": "active",
    "active_mission": {
      "mission_id": "goto-001",
      "type": "goto",
      "frame_id": "global_map",
      "x_m": 20.0,
      "y_m": 30.0,
      "submitted_seq": 5,
      "subgoal_index": 0,
      "subgoal_count": 1,
      "current_goal": {
        "frame_id": "global_map",
        "x_m": 20.0,
        "y_m": 30.0
      }
    },
    "mission_queue": {
      "size": 1,
      "capacity": 16,
      "mission_ids": ["goto-002"]
    },
    "manual_setpoint_active": false,
    "coordination": {
      "state": "waiting",
      "reason": "corridor_lease",
      "priority_owner_vehicle_id": "mock_vehicle_02"
    },
    "navigation": {
      "status": "active",
      "goal_mode": "exact",
      "planning": false,
      "algorithm": "d_star_lite",
      "replan_count": 1
    }
  },
  "safety": {
    "state": "clear",
    "reason": null,
    "obstacle_clearance_m": null,
    "edge_clearance_m": null
  },
  "localization": {
    "frame_id": "anchor_map",
    "anchor_id": "mock_vehicle_01_anchor",
    "x_m": 2.1,
    "y_m": -1.5,
    "yaw_rad": 0.4,
    "covariance_diagonal": [0.0, 0.0, 0.0],
    "quality": "nominal",
    "timestamp_s": 1717800000.4,
    "revision": 12,
    "local_map_revision": 8
  },
  "p2p_map_sync": {
    "enabled": true,
    "ready": true,
    "peer_id": "12D3KooW...",
    "connected_vehicle_ids": ["mock_vehicle_02", "mock_vehicle_03", "mock_vehicle_04"],
    "own_known_cells": 128,
    "own_dirty_cells": 0,
    "published_deltas": 4,
    "received_deltas": 9,
    "rejected_deltas": 0,
    "publish_failures": 0,
    "sequence_gaps": 0,
    "published_peer_states": 12,
    "received_peer_states": 18,
    "rejected_peer_states": 0,
    "peer_state_publish_failures": 0,
    "active_peer_vehicle_states": 3,
    "published_motion_intents": 12,
    "received_motion_intents": 18,
    "rejected_motion_intents": 0,
    "motion_intent_publish_failures": 0,
    "active_peer_motion_intents": 3,
    "peer_sources": {
      "mock_vehicle_02": {"map_epoch": 1, "last_sequence": 3, "known_cells": 72}
    },
    "collaborative_evidence_cells": 200,
    "collaborative_view_current": false,
    "collaborative_known_cells": null
  }
}
```

`pose` 不泄露绝对仿真真值。`controller` 是模式、队列和导航的权威快照。
`localization.quality` 为 `nominal`、`degraded` 或 `lost`。
`controller.coordination.state` 为 `idle`、`tentative`、`reserved` 或 `waiting`：
`tentative` 表示本车已声明走廊但尚未得到对端确认，`reserved` 表示本车是唯一已确认
owner，`waiting` 的 `reason` 还可为 `corridor_election`、`corridor_lease`、
`reservation_sync`（expected peer state/intent 不完整或同源 generation 不一致）或
`space_time_reservation`
（motion-intent v4 时间窗尚不可进入）。已知胜者通过 `priority_owner_vehicle_id` 指出；
同步等待可能为 `null`。该摘要只用于观察；客户端不得据此绕过 `RobotController` 或
LocalSafety 下发运动。
没有启用场景级 P2P 时，`p2p_map_sync` 为 `{"enabled":false}`。协同摘要仅供观察，
远端地图当前不会改变本车 D* Lite 或安全控制。peer state 与 motion intent 的
published/received/rejected/failure 计数及当前有效租约数分别由同名字段报告。常规遥测
不会物化完整协同地图：
`collaborative_evidence_cells` 是各来源证据格数量之和；只有调用显式协同视图查询后，
`collaborative_view_current` 才为 `true`，此时 `collaborative_known_cells` 才是去重投影后的
精确数量，否则为 `null`。

### scan

紧随 `pose` 的 scan 使用相同 `seq` 和 `timestamp_s`：

```json
{
  "type": "scan",
  "timestamp_s": 1717800000.4,
  "seq": 12,
  "frame_id": "laser",
  "config": {
    "model": "ydlidar_tmini",
    "min_angle": 0.0,
    "max_angle": 6.273765,
    "angle_increment": 0.00942,
    "scan_time": 0.166667,
    "min_range": 0.02,
    "max_range": 12.0,
    "point_count": 667,
    "angle_unit": "rad",
    "range_unit": "m",
    "angle_direction": "clockwise_from_forward",
    "no_return": {"range": 0.0, "intensity": 0.0}
  },
  "points": [
    {"angle": 0.0, "range": 2.5, "intensity": 1.0},
    {"angle": 0.00942, "range": 0.0, "intensity": 0.0}
  ]
}
```

有效墙体回波按 `0.01 m` 量化；无回波为 `range=0`、`intensity=0`，不能解释为
障碍物。水平 Tmini 不检测落差；模拟器用独立下视输入生成 edge safety 和
Forbidden 证据。

## 协议错误

```json
{
  "type": "error",
  "timestamp_s": 1717800000.5,
  "seq": 9,
  "code": "invalid_fields",
  "message": "command has missing or unexpected fields"
}
```

协议错误会先触发故障制动；若自动任务存在，会额外收到 `paused` 事件，原因为
`invalid_command`。常见 code：

```text
invalid_json_text, message_too_large, invalid_json, invalid_message
missing_seq, invalid_seq, stale_seq
invalid_type, invalid_action, invalid_fields, invalid_number
drive_out_of_range, invalid_missions, mission_batch_too_large
duplicate_mission_id, invalid_mission, invalid_mission_type, goal_out_of_range
```
