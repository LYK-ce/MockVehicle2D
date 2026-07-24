# WebSocket 通信协议

## 基本信息

| 项目 | 值 |
|------|------|
| 传输协议 | WebSocket |
| 数据格式 | JSON 文本消息 |
| 编码 | UTF-8 |
| 角色 | 小车 = Server，PC = Client |
| 默认端口 | 9090 |

除兼容旧客户端的精确 `{"cmd":"..."}` 格式外，每条消息均为 JSON 文本对象，
顶层必有 `type` 字段。

坐标系：2D 用 `(x, y)`，3D 高度用 `z`，与 Godot 坐标系统一。

---

## 连接流程

连接分两层：

| 阶段 | 触发条件 | 含义 |
|------|---------|------|
| WebSocket 握手完成 | TCP 升级为 WS | 物理通道建立 |
| `hello` 包收到 | 小车发送身份 | **正式建立连接** |

Pictor 仅在收到 `hello` 后才认为连接可用，之后才开始处理 `pose`、`map_*` 等业务消息。
`hello` 之前收到的任何消息将被丢弃。
同一车辆同时只接受一个控制连接；已有控制器时，新连接只收到
`{"type":"error","code":"vehicle_busy",...}` 并结束，不会收到 `hello`，也不会改变车辆状态。
当前控制器断开会安全停车并释放控制权，随后连接继续沿用已有 odometry 和本地地图 revision。

```
小车 ── TCP 握手 ──→ PC       (物理层)
小车 ── hello ──→ PC          ← 必须第一帧，业务层连接建立
小车 ── map_full ──→ PC
小车 ── pose ──→ PC
```

---

## 上行：小车 → PC

### hello — 注册身份

连接建立后立即发送，声明车辆 ID。

```json
{
    "type": "hello",
    "vehicle_id": "car_0"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `vehicle_id` | string | 车辆唯一标识 |

### pose — 车辆位姿

实时发送车辆位置、朝向和速度。

```json
{
    "type": "pose",
    "ts": 1717800000.123,
    "seq": 12,
    "source": "anchored_odometry",
    "x": 1.5,
    "y": 3.2,
    "z": 0.0,
    "yaw": 0.785,
    "vx": 0.5,
    "vy": 0.0,
    "omega": 0.0,
    "collision": false,
    "command": "forward",
    "localization": {
        "frame_id": "anchor_map",
        "anchor_id": "mock_vehicle_01_anchor",
        "x_m": 1.5,
        "y_m": 3.2,
        "yaw_rad": 0.785,
        "covariance_diagonal": [0.0, 0.0, 0.0],
        "quality": "nominal",
        "timestamp": 123.0,
        "revision": 12
    },
    "control_mode": "autonomous",
    "navigation": {
        "status": "active",
        "goal": {"x_m": 8.0, "y_m": 3.5},
        "reason": null
    },
    "safety": {
        "state": "limited",
        "reason": "safety_obstacle",
        "obstacle_clearance_m": 0.7,
        "edge_clearance_m": null
    }
}
```

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `ts` | f64 | 秒 | Unix 时间戳 |
| `seq` | u64 | — | 遥测帧序号；紧随其后的 `scan` 使用相同值 |
| `source` | string | — | 固定为 `anchored_odometry` |
| `x`, `y` | f32 | 米 | 锚定里程计投影到 `global_map` 的 2D 坐标 |
| `z` | f32 | 米 | 高度 |
| `yaw` | f32 | 弧度 | 偏航角 |
| `vx`, `vy` | f32 | 米/秒 | 2D 速度分量 |
| `omega` | f32 | 弧度/秒 | 实际角速度；左旋为负，右旋为正 |
| `collision` | bool | — | 最近一次平移是否被碰撞截停 |
| `command` | string | — | 当前有效命令；看门狗超时后为 `stop` |
| `localization` | object | — | `anchor_map` 中的局部位姿、协方差对角线、质量和 revision |
| `control_mode` | string | — | `navigation.status=active` 时为 `autonomous`，否则为 `manual` |
| `navigation` | object | — | `status`、当前或最后一个 `goal`，以及终止 `reason` |
| `safety` | object | — | 最新安全状态、原因、前进方向障碍净空和边缘净空；未发现时净空为 `null` |

`pose` 不再发送绝对仿真真值。小车只保存已知出生锚点
`global_map → anchor_map`，之后通过物理运动增量更新锚定 odometry。默认零噪声时显示效果
与旧协议一致；配置噪声后会产生可重放漂移。该估计不是 SLAM，也不保证长期无漂移。
`localization.quality` 为 `nominal`、`degraded` 或 `lost`；`lost` 会阻止自动任务和本地地图写入。
`navigation.status` 为 `idle`、`active`、`reached`、`blocked` 或 `cancelled`；结束后仍保留目标与原因，便于客户端确认结果。

---

### scan — 本地二维激光扫描

每次 `pose` 后发送一帧 **YDLidar Tmini** 风格的局部扫描；Server 按单调时钟以名义 6 Hz 发送这对消息。它只表示小车相对坐标系中的障碍物距离，**不**提供定位、点云、SLAM 或全局地图。

```json
{
    "type": "scan",
    "ts": 1717800000.123,
    "seq": 12,
    "frame_id": "laser",
    "config": {
        "min_angle": 0.0,
        "max_angle": 6.273765,
        "angle_increment": 0.00942,
        "time_increment": 0.000249875,
        "scan_time": 0.166667,
        "min_range": 0.02,
        "max_range": 12.0,
        "point_count": 667,
        "model": "ydlidar_tmini",
        "range_sample_rate_hz": 4000,
        "scan_rate_hz": 6,
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

`angle` 的单位是弧度，`range` 的单位是米，字段形式与 YDLidar SDK 的 `LaserPoint` 对齐。`yaw=0`、`angle=0` 指向世界 `+x`；在本模拟器的 `+y` 向下栅格中，正角度朝 `+y` 增长（从上方看顺时针）。

默认配置模拟 Tmini 的 `ydlidar_tmini` 元数据：360°、0.02–12 m、名义 4000 Hz 测距、名义 6 Hz 扫描、667 个均匀射线。`667 = round(4000 / 6)`，为保持每帧固定点数，`time_increment = scan_time / 667`（约 4002 Hz）；不会实现复杂的每帧交替点数。有效墙体回波按 0.01 m 确定性量化，强度固定为 `1.0`。无回波统一编码为 `range: 0.0, intensity: 0.0`，包括最大量程外、离开已知栅格或未知空间；它**不是**障碍物。扫描没有噪声或反射率模型。

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | string | 固定为小车本地 `laser` 坐标系 |
| `seq` | u64 | 与同一状态快照的 `pose.seq` 相同 |
| `config` | object | 一帧的 LaserScan 配置与无回波约定 |
| `points` | array | 按角度递增排列的 `{angle, range, intensity}` 读数 |

### map_full — 全量地图

连接建立或重连后，在 `hello` 之后发送完整地图。

```json
{
    "type": "map_full",
    "ts": 1717800000.200,
    "source": "simulator_ground_truth",
    "voxels": [
        {"gx": 0, "gy": 0, "gz": 0, "state": 0, "conf": 0.95},
        {"gx": 1, "gy": 0, "gz": 0, "state": 1, "conf": 0.80}
    ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts` | f64 | Unix 时间戳 |
| `source` | string | 固定为 `simulator_ground_truth`；不是车辆传感器输出 |
| `voxels` | array | 全量体素列表 |
| `gx`, `gy` | i32 | 2D 网格坐标 |
| `gz` | i32 | 高度层 |
| `state` | u8 | `0`=可通行，`1`=墙/障碍物，`2`=无地面/落差 |
| `conf` | f32 | 置信度 0.0~1.0 |

### map_delta — 增量地图（尚未实现）

仅发送变化的格子。

```json
{
    "type": "map_delta",
    "ts": 1717800000.300,
    "voxels": [
        {"gx": 2, "gy": 1, "gz": 0, "state": 1, "conf": 0.90}
    ]
}
```

字段同 `map_full`。

---

## 下行：PC → 小车

### cmd — 控制命令

规范格式：

```json
{
    "type": "cmd",
    "seq": 42,
    "cmd": "forward"
}
```

`seq` 必须是非负整数。为兼容旧客户端，也接受字段严格为 `{"cmd":"forward"}` 的 legacy 格式，其确认消息中 `seq` 为 `null`。二进制帧、非 JSON、非对象、错误 `type`、额外字段、非法 `seq` 或未知 `cmd` 都会立即停车并返回 `error`。

| 命令 | 说明 |
|------|------|
| `forward` | 前进 |
| `forward_left` | 前进并左转（W+A） |
| `forward_right` | 前进并右转（W+D） |
| `backward` | 后退 |
| `backward_left` | 后退并左转（S+A） |
| `backward_right` | 后退并右转（S+D） |
| `spin_left` | 左旋 |
| `spin_right` | 右旋 |
| `stop` | 停止（松手时发送） |

默认运动参数：前进/后退 `±0.5 m/s`；在 `+y` 向下、正 yaw 顺时针的坐标约定中，左旋/右旋为 `∓90°/s`。组合命令同时应用对应线速度与角速度，形成弧线。连续 `1.0 s` 未收到有效非停止命令时，看门狗令车辆停止。这些值可通过 `mockvehicle2d serve` 参数校准。

### drive — 连续速度控制

```json
{
    "type": "drive",
    "seq": 43,
    "linear_mps": 0.25,
    "angular_rps": -0.4
}
```

`linear_mps` 和 `angular_rps` 必须是有限 JSON 数字，布尔值不算数字；绝对值分别不得超过 Server 的 `--linear-speed` 和 `--angular-speed` 配置。字段必须严格为示例中的四项。`0, 0` 等价于停车；非零速度沿用与 `cmd` 相同的看门狗、运动积分和碰撞停车。确认仍使用 `cmd_ack`，其中 `cmd` 为 `"drive"`。

### goto — 全局锚定目标

```json
{
    "type": "goto",
    "seq": 44,
    "x_m": 12.0,
    "y_m": 8.5
}
```

`x_m`、`y_m` 是 `global_map` 中的有限 JSON 数字，字段必须严格为以上四项。Server 用已知
出生锚点把目标转换到 `anchor_map`，控制器只读取锚定里程计，先对准目标，再沿直线前进并
在接近目标时减速；它不做 A* 或绕障。定位 `degraded` 时自动线速限制为一半；
`lost` 时 `goto_ack.accepted=false` 或活动任务变为 `blocked`，原因为
`localization_lost`；未结算的旧自动运动不会再积分，但新的人工 `cmd` / `drive` 仍可接管。
碰撞、安全硬停止或安全输入故障也会立即停车且不会自行重启。

新的 `goto` 会替换旧目标。任何 `cmd` 或 `drive`（包括 `stop`），以及任何非法输入，都会永久取消当前目标；除非客户端重新发送 `goto`，否则不会恢复自动行驶。连接断开也会停车并取消活动目标。

确认消息独立为：

```json
{
    "type": "goto_ack",
    "ts": 1717800000.400,
    "seq": 44,
    "goal": {"x_m": 12.0, "y_m": 8.5},
    "accepted": true
}
```

`accepted` 只有在目标已进入 `active` 状态时才为 `true`。若旧运动在命令交接时
已因碰撞或安全门控停止，则返回 `accepted: false`，并以 `reason` 报告
`collision`、`safety_obstacle`、`safety_edge`、`safety_sensor_fault` 或
`localization_lost`。

### cmd_ack — 命令确认

```json
{
    "type": "cmd_ack",
    "ts": 1717800000.400,
    "seq": 42,
    "cmd": "forward",
    "accepted": true
}
```

`accepted` 只有在命令通过碰撞与安全门控并实际安装后才为 `true`。若命令被拒绝，
Server 返回 `accepted: false`，并用 `reason` 报告 `collision`、
`safety_obstacle`、`safety_edge` 或 `safety_sensor_fault`。

### error — 命令错误

```json
{
    "type": "error",
    "ts": 1717800000.500,
    "seq": 42,
    "code": "invalid_cmd",
    "message": "unsupported cmd"
}
```

无法安全提取序号时 `seq` 为 `null`。错误输入总是先触发安全停车。
`vehicle_busy` 是连接级拒绝，不会停车、取消或推进当前控制器的车辆。

---

## 消息一览

```
上行 (小车 → PC)          下行 (PC → 小车)
─────────────────         ─────────────────
hello                      cmd / drive / goto
map_full
pose
scan
cmd_ack / goto_ack / error
map_delta
```

`map_full` 是 `simulator_ground_truth`，仅供物理、传感器生成和调试显示；`pose` 是
`anchored_odometry`，`scan` 是模拟的 Tmini 本地观测。扫描会累计到车辆内存中的
`ObservedGrid`，其 Unknown/Free/Occupied、revision 和 delta 尚未通过 WebSocket 上传。
车辆 runtime 跨控制器断开重连保留。当前没有 SLAM、scan matching、回环、D* Lite 或中央
地图同步。

## 安全运行时

水平安装的 Tmini 只对 `state=1` 的墙产生正距离回波；`range=0` 仍表示无回波，不能解释为障碍物。Tmini 不能检测地面落差，因此 `state=2` 和地图越界由模拟器独立的向下地面探测辅助量判断，不能宣称来自雷达。

运行时以车辆圆形 footprint 为边界输出障碍/边缘净空：Tmini 正回波投影到行驶方向的完整圆形扫掠走廊，不使用固定角度扇区。净空 `<=0.25 m` 时硬停止，自动模式在 `0.25–1.0 m` 内线性降低平移速度，手动模式不执行慢速区降速但仍执行硬停止。即使事件循环延迟，旧速度也只按不超过 `0.05 m` 且不越过硬停止净空的小步推进，并在每步前重新观测。带平移的手动命令触发硬停止后会全部停车；客户端可发送新的反向或纯旋转命令重新评估并脱困。停车状态不会自行恢复。

`safety.state` 为 `clear`、`limited`、`stopped` 或 `fault`；对应原因目前为 `safety_obstacle`、`safety_edge` 或 `safety_sensor_fault`。安全输入故障采用 fail-safe：自动任务变为 `blocked`，手动命令也全部停车。未发现正雷达回波或有限前视范围内未发现边缘仅表示当前观测为 clear，不表示更远区域已知。生产模拟默认输入健康；`healthy=false` 只用于确定性故障测试。

车辆自有地图按每条射线把命中前区域标为 Free、命中格标为 Occupied、遮挡后保持 Unknown。
当前模拟协议把 `range=0` 的无回波射线更新到最大量程 Free；真实 Tmini 的无回波原因更复杂，
接入硬件前必须校准，不能直接沿用。水平 Tmini 仍不能检测跌落。
