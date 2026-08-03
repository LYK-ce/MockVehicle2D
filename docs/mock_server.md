# Robot Controller 模拟架构

## 职责边界

MockVehicle2D 对齐真实小车的控制分层，但保留仿真环境和确定性传感器：

```text
Client
  │ WebSocket v4
  ▼
Protocol
  │ typed Command
  ▼
RobotController
  ├── OpMode: Manual / Auto
  ├── Manual lease
  ├── active mission
  ├── bounded pending queue
  └── mission lifecycle
        │
        ├── Manual desired velocity
        └── GotoController desired velocity
                        │
                        ▼
                LocalSafetyRuntime
                        │
                        ▼
                     Vehicle
```

| 模块 | 负责 | 不负责 |
|------|------|--------|
| `protocol.py` | 严格 JSON、字段/范围校验、类型化命令 | 模式和任务状态 |
| `controller.py` | 模式、队列、生命周期、速度输出所有权 | WebSocket、真值地图生成 |
| `navigation.py` | 根据局部状态生成自动期望速度 | 直接修改车辆 |
| `safety.py` | 手动/自动安全门控、故障停车、安全推进 | 任务队列 |
| `server.py` | 独占连接、帧调度、遥测和事件发送 | 自主决策 |
| `vehicle.py` | 运动学、执行器设定值、看门狗 | 任务和协议 |

安全运行时和碰撞检查可以直接停车，因为它们是控制器之外的独立故障保护；除此之外，
业务路径不得绕过 `RobotController` 安装运动设定值。

## 状态模型

```text
OpMode
  Manual
  Auto

AutoState
  Idle      没有任务
  Active    正在启动、规划或执行当前任务
  Paused    当前任务和队列保留，但不输出自动速度
  Blocked   当前任务失败，等待 resume 重试或 cancel_all
```

队列由一个活动任务和一个有界待执行队列组成。到达后自动取下一个任务；阻断后不会跳过
当前任务。`resume` 会从当前定位和地图重新启动活动任务。`cancel_all` 清除活动任务和所有
待执行任务。

模式切换规则：

| 操作 | 结果 |
|------|------|
| Manual → Auto（无任务） | `Auto/Idle` |
| Manual → Auto（有保留任务） | `Auto/Paused`，需 `resume` |
| Auto → Manual | 先停车，活动任务转为 paused，队列保留 |
| 重复切换到当前模式 | 幂等，无额外停车和事件 |
| 任意模式 → `stop_motion` | 立即停车并清除手动租约；Auto 任务暂停保留 |
| 控制连接断开 | 停车；自动任务暂停；释放独占连接 |
| 非法协议输入 | 停车；自动任务暂停，原因 `invalid_command` |

在 `Auto/Idle` 且没有活动或排队任务时，`pause` 保持 Idle，不产生虚假的预暂停状态。

## 一帧的执行顺序

Server 以 Tmini 名义扫描周期（约 6 Hz）执行：

1. 安全运行时把上一设定值推进到当前单调时间；
2. 从仿真真值环境生成当前 Tmini scan；
3. 用真值运动增量更新模拟 odometry；
4. 用旧占据证据做有限窗 scan matching；
5. 将当前 scan 写入车辆自有 `ObservedGrid`，得到 map delta；
6. `RobotController.tick()` 处理任务、规划和期望速度；
7. 期望速度通过安全门控后安装到 `Vehicle`，下一周期生效；
8. 发送同一帧的 `pose`、`scan` 和任务事件。

命令到达时，Runtime 先把旧设定值安全推进到接收时刻，再执行命令。这样命令处理和遥测
都不会倒退模拟时间。

`--realtime-factor` 只缩放上述循环的墙钟等待，默认值为 `5`。线/角速度、100 ms fleet
tick、Tmini 扫描周期、命令超时和安全阈值仍以模拟秒及 SI 单位计算；传入 `1` 可按实时
速度运行。

## 手动控制

手动控制直接给出 `linear_mps` 与 `angular_rps`，但仍经过安全门控。非零设定值只在
`command-timeout-s` 租约内有效；客户端长按时应重复发送 `drive`，松手发送 `stop`。
看门狗到期、碰撞、边缘/障碍硬停止或安全输入故障都会停车。

控制端不能假设自己掌握最新模式来完成安全停车。`mode/stop_motion` 是模式无关的停车
入口：它在 Manual 清除速度租约，在 Auto 暂停并保留任务；重复调用保持幂等。

## 自动控制

Auto `push` 只写入父任务队列。`goto` 直接提供一个目标；`patrol` 和矩形 `coverage`
确定性生成最多 1024 个子目标，并按序复用同一个 `GotoController`。控制帧将当前
`global_map` 子目标通过出生锚点转换为 `anchor_map`，再由有限视野 D* Lite 规划。
`GotoController` 返回期望速度，不直接控制车辆。中间子目标不成为独立任务，父任务的
到达、阻断、暂停和取消通过 `mission_update` 明确发布。

同一 `mission_id` 和完全相同任务定义可用新的 `seq` 安全重试；不会生成第二个任务。
相同 ID 对应不同类型、航点、轮次、区域或间距会返回 `mission_id_conflict`。该记录在
Server 进程生命周期内永久保留，不会因任务数量增加而静默淘汰；进程重启会清空，
因为模拟器当前不提供持久化。
Auto 已为 Active 时重复 `resume` 也是幂等操作，不会重启 D* 搜索。

## 连接与故障语义

- 一个 Runtime 同时只允许一个控制 WebSocket；其他连接收到 `vehicle_busy`。
- `hello` 是业务首帧，声明协议版本、控制租约、任务坐标系、地图元数据和控制器快照。
- `RobotController` 先把任务状态变化写入进程内 event ledger，再由 Server 发送。
  每个事件有 `event_epoch` 和严格递增的 `event_seq`；只有 WebSocket `send` 成功后，
  本连接游标才前移。
- 新连接在 `hello` 后自动按序重放本进程的全部任务事件。重连可能产生重复传输，客户端
  按稳定的 `(event_epoch, event_seq)` 幂等去重；事件不会因 pose、scan、ack 或事件
  发送失败而丢失。Server 进程重启后 epoch 会变化。
- JSON 发送和二进制地图 chunk 均有超时，慢客户端不会无限占用控制循环。
- 连接断开不删除 odometry、本地地图、活动任务或队列；车辆停车并把自动状态设为
  `Paused`。
- 重连后客户端从 `hello.controller` 恢复视图，并显式发送
  `switch_to_auto`（如需要）和 `resume`。

## 真值与局部状态

`MapGrid` 是模拟世界真值，只服务碰撞、传感器生成和调试 `map_full`。自主导航只消费：

```text
AnchorSpec
PoseEstimate
ObservedGrid
LocalMapDelta
SafetyObservation
```

因此隐藏障碍在雷达观测前保持 Unknown；进入视野并写入 Occupied 后才触发 D* Lite
重规划。当前轻量 scan matching 不包含回环、位姿图或全局优化。

## 启动

```bash
mockvehicle2d serve \
  --port 19090 \
  --vehicle-id mock_vehicle_01 \
  --realtime-factor 5 \
  --mission-capacity 16 \
  --linear-speed-mps 0.5 \
  --angular-speed-rps 1.5708 \
  --vehicle-radius-m 0.5 \
  --command-timeout-s 1.0
```

Pictor 或其他客户端必须实现
[WebSocket v4 协议](websocket_protocol.md)。
