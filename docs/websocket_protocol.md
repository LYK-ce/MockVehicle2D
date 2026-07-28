# WebSocket v4 协议

## 基本约定

| 项目 | 值 |
|------|----|
| 角色 | 小车是 WebSocket Server，控制端是 Client |
| 默认地址 | `ws://127.0.0.1:19090` |
| 命令格式 | UTF-8 JSON 文本对象 |
| 协议版本 | `4` |
| 控制权 | 每辆车同时只有一个独占连接 |
| 单位 | m、s、rad、m/s、rad/s |
| 任务坐标系 | `global_map` |

下行命令的顶层字段必须与定义完全一致，不允许缺失或额外字段。JSON 重复 key、
`NaN`、`Infinity`、二进制命令、超过 64 KiB 的消息均无效。每个连接内的 `seq` 是
无符号 64 位整数且必须严格递增。

旧的 `cmd`、`drive`、直接 `goto`、`nl_command` 和无 `type/seq` 格式不属于 v4。

## 连接流程

```text
Client ── WebSocket handshake ──► Vehicle
Client ◄──────── hello ────────── Vehicle
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
  "ts": 1717800000.1,
  "seq": null,
  "code": "vehicle_busy",
  "message": "another controller owns the vehicle lease"
}
```

断开会立即停车并释放租约。odometry、本地地图、活动任务和待执行队列保留；自动状态
变为 `paused`。重连后由 `hello.controller` 恢复 UI，再显式 `resume`。

## 下行命令

### 模式

```json
{"type":"mode","seq":1,"action":"switch_to_auto"}
{"type":"mode","seq":2,"action":"switch_to_manual"}
{"type":"mode","seq":3,"action":"stop_motion"}
```

模式命令不受当前模式限制。实际切换先停车；重复切到当前模式是幂等操作。
Auto → Manual 暂停并保留任务。Manual → Auto 若有保留任务仍停在 `paused`，需要
`auto/resume`。

`stop_motion` 是模式无关的安全停车入口：

- 在 Manual 中立即停车并清除手动速度租约，模式仍为 Manual；
- 在 Auto 中立即停车，活动任务和队列保留，状态变为 `paused`；
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
      "mission_id": "goto-002",
      "type": "goto",
      "frame_id": "global_map",
      "x_m": 24.0,
      "y_m": 35.0
    }
  ]
}
```

仅在 `auto` 模式有效。当前只支持 `goto`。`mission_id` 为 1–64 个 ASCII 字母、
数字、点、下划线、冒号或连字符。坐标必须有限且绝对值不超过 `1,000,000 m`。

一次 push 是原子的：

- batch 不能为空、不能超过任务队列配置，且 batch 内 ID 必须唯一；
- 待执行队列空间不足时整个 batch 拒绝；
- 已知 `mission_id` 携带相同 `frame_id/x_m/y_m` 时视为幂等重试，不重复入队；
- 已知 `mission_id` 携带不同目标时返回 `mission_id_conflict`。

客户端重试时必须使用新的、更大的命令 `seq`，但沿用原 `mission_id`。

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

## 命令确认

每个通过语法边界的命令返回一个 `command_ack`：

```json
{
  "type": "command_ack",
  "timestamp_s": 1717800000.2,
  "ts": 1717800000.2,
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
  "timestamp_s": 1717800000.3,
  "ts": 1717800000.3,
  "mission_id": "goto-001",
  "submitted_seq": 5,
  "status": "blocked",
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
    "auto_state": "idle"
  }
}
```

`map.source=simulator_ground_truth` 表示调试真值，不能作为自动规划输入。

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
  "ts": 1717800000.4,
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
      "submitted_seq": 5
    },
    "mission_queue": {
      "size": 1,
      "capacity": 16,
      "mission_ids": ["goto-002"]
    },
    "manual_setpoint_active": false,
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
  }
}
```

`pose` 不泄露绝对仿真真值。`controller` 是模式、队列和导航的权威快照。
`localization.quality` 为 `nominal`、`degraded` 或 `lost`。

### scan

紧随 `pose` 的 scan 使用相同 `seq` 和 `timestamp_s`：

```json
{
  "type": "scan",
  "timestamp_s": 1717800000.4,
  "ts": 1717800000.4,
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
  "ts": 1717800000.5,
  "seq": 9,
  "code": "invalid_fields",
  "message": "command has missing or unexpected fields"
}
```

协议错误会先触发故障停车；若自动任务存在，会额外收到 `paused` 事件，原因为
`invalid_command`。常见 code：

```text
invalid_json_text, message_too_large, invalid_json, invalid_message
missing_seq, invalid_seq, stale_seq
invalid_type, invalid_action, invalid_fields, invalid_number
drive_out_of_range, invalid_missions, mission_batch_too_large
duplicate_mission_id, invalid_mission, invalid_mission_type, goal_out_of_range
```

可选的 `mockvehicle2d.instruction.dispatcher` 可以把已校验意图转换为单条合法 v4
命令，但它不是 WebSocket 消息类型。Server 不接受旧 `nl_command`，翻译器也不会
隐式切换模式或绕过 `RobotController`。
