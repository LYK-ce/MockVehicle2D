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
| `vehicle.py` | 有界加减速、运动学、执行器设定值、看门狗 | 任务和协议 |

普通安全停车与看门狗同样只请求有界制动。静态碰撞和多车同时仲裁可以立即钳制实际
速度，因为它们必须拒绝已判定不安全的物理候选轨迹；除此之外，业务路径不得绕过
`RobotController` 安装运动设定值。

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
| Auto → Manual | 先请求制动，活动任务转为 paused，队列保留 |
| 重复切换到当前模式 | 幂等，无额外停车和事件 |
| 任意模式 → `stop_motion` | 请求有界制动并清除手动租约；Auto 任务暂停保留 |
| 控制连接断开 | 请求有界制动；自动任务暂停；释放独占连接 |
| 非法协议输入 | 请求有界制动；自动任务暂停，原因 `invalid_command` |

在 `Auto/Idle` 且没有活动或排队任务时，`pause` 保持 Idle，不产生虚假的预暂停状态。

`Vehicle` 分开保存 target 和 executed linear/angular velocity。每次物理推进用配置的
`1 m/s²` 线加速、`1 m/s²` 线减速和 `π rad/s²` 角加速上限逼近 target；反向命令先制动
到零再向相反方向加速。遥测速度和 P2P 车辆状态报告 executed velocity，WebSocket 速度
命令仍表示 target，不改变 v4 JSON schema。realtime factor 只缩短墙钟等待，不参与该
动力学计算。

每个安全物理小步的停车净空至少包含固定 `0.25 m` 余量、当前小步位移上界和按配置线
减速度计算的 `v²/(2a)` 制动距离。正反向切换或 target 已归零但车辆仍在制动时，风险
感知继续使用 executed velocity 的实际运动方向；`serve` 与 `fleet` 复用同一准备逻辑。

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

### 多车局部协调

多车节点通过 peer-state v1 发布 executed pose/velocity，通过 motion-intent v4 发布约
`4 s` 的相对 cell 时间窗、等待/任务年龄、`0.8 s` 短 commit、goal hold 和可选的有向
corridor descriptor。v4 还携带必需的 `vacate_request`：通常为 `null`；当高优先级车辆的
OwnMap 路线被已完成任务的 Auto/Idle peer 截断时，可定向携带目标车辆、其 footprint cell
及 `2..64` 格的 OwnMap 路线窗口。它们都来自车辆 odometry 和局部规划，不携带仿真真值；
收到的 peer map evidence 也不会进入 OwnMap 或自主规划。接收端按 receipt time 重建 reservation，
因此不要求车辆时钟同步。

通用路径沿 OwnMap D* 空间候选运行 prioritized SIPP，覆盖 vertex、edge swap、同步几何
交叉和 goal hold。未提交候选按冲突等待年龄及 owner/vehicle ID 排序，已 commit 的冲突
声明只使用 owner/vehicle ID 全序；未同步的 `task_age_ticks` 仅供观测，未来需 Lamport task
token 才能表达跨车任务先后。占路链递归传播 owner 并只尝试一个邻格 detour。expected
peer 的 state/intent 缺失、过期、同源 generation 不一致或 sidecar 未 ready 时新前缀
fail-closed。相邻不同 cell 的时间窗必须有正 travel time，commit 上限严格为
`min(0.8 s, trajectory 最后 leave_offset_s)`；LocalSafety 和同步物理碰撞仲裁仍是最终
保障。OwnMap
能够确认完全观测、直线且内宽不超过约 `3 m` 的瓶颈时，再对重叠 corridor 做去中心化
选举和确认：模式外停车不受影响；进入前只允许一个 confirmed owner；出口侧反向队列仅
front waiter 提前做可逆侧移，rear waiter 原地等待；ACK 后 owner 仍要等 front 的连续位姿
和剩余侧移段离开两车 footprint 与 `0.3 m` 安全包络，才可跨 entry。owner 通过远端边界
后，车体与安全余量都清空才释放。租约结束后若 saved rejoin 段已被 live peer 占据，waiter
从侧袋位置交回本地导航重规划。已归因 peer 暂时切断 D* 路径时任务保持 active；匿名动态
遮挡只有一次有界 restart grace，持续或闪烁阻塞仍会终止为 `no_path`。

收到定向请求的空闲车辆只通过真实 Goto、同一 SIPP 与 LocalSafety 侧移和返回，不创建
内部 Mission。fresh 的显式 `vacate_request:null` 需要连续 3 tick 的 clear pose/trajectory
才允许返回；请求缺失、过期、身份/generation 不一致时停车并保持 fail-closed。

这不是中央调度器，也不声称实现完整 PIBT/MAPF。第一版 SIPP 只给一条 D* 候选和一个
邻格 detour 排时，也没有独立网络 propose/ACK/commit 往返。当前走廊严格一次放行一辆车，
长走廊和深队列吞吐近似线性；ready-owner skipping 和同向 batching/convoy 尚未实现。
详细检测、重规划和释放语义见 [有限视野寻路](pathfinding.md)。

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
  --linear-acceleration-mps2 1.0 \
  --linear-deceleration-mps2 1.0 \
  --angular-acceleration-rps2 3.1416 \
  --vehicle-radius-m 0.5 \
  --command-timeout-s 1.0
```

Pictor 或其他客户端必须实现
[WebSocket v4 协议](websocket_protocol.md)。
