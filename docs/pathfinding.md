# 有限视野寻路

Auto 队列中的 `goto` 任务使用车辆自己的有限视野占据栅格和 D* Lite。模拟器完整真值
地图不参与路径规划，只用于物理碰撞、传感器生成和调试显示。

## 运行链路

```text
Tmini scan + anchored odometry
              │
              ▼
bounded scan matching（只匹配旧 Occupied 证据）
              │
              ▼
ObservedGrid: Unknown / Free / Occupied + delta
              │
              ▼
D* Lite（增量修复路径）──► 约 4 s 空间候选
                                  │
                                  ▼
                    prioritized SIPP + peer reservation
                                  │ 只提交约 0.8 s
                                  ▼
                            局部 waypoint ──► 期望速度
                                                    │
RobotController ────────────────────────────────────┤
                                                    ▼
                                                  safety
```

一帧按固定顺序执行：

1. 安全运行时用上一周期速度推进到当前时间。
2. 在同一时刻采集一帧 Tmini scan，并用运动增量预测位姿。
3. 扫描与旧地图配准；只在质量门槛通过时修正位姿。
4. 用修正后的位姿写入扫描，生成地图 delta。
5. D* Lite 消费 delta 并为下一周期选择速度。

`pose` 与紧随其后的 `scan` 使用相同 `seq` 和 `timestamp_s`。

每条 `goto`、`patrol` 或 `coverage` 命令只进入接收该命令的车辆控制器；这里没有车队任务
分配器。多车运动协调只临时改变期望速度、局部绕行点和路权等待，不会交换或静默取消
任务。唯一的任务展开例外是显式设置 `coverage.coordination_id`：任务首次激活时，各车从
排序后的 `{本车} + 固定 expected peer allowlist` 独立得到同一成员顺序，并沿原矩形长轴
选择自己的连续子矩形。所有成员必须收到相同 ID、区域和间距；部分下发、动态成员与
运行中重分配暂不支持。省略该字段时 Patrol/Coverage 的既有本车展开完全不变。

## D* Lite

`DStarLitePlanner` 是实际 `goto` 运行时的增量规划器，并非反复调用 A* 的包装。

| 项目 | 当前语义 |
|------|----------|
| 连通性 | 八连通 |
| Cardinal / diagonal 代价 | `1 m` / `√2 m` 乘目标格状态代价 |
| Free 代价 | `1` |
| Unknown 代价 | 默认 `3`，可通行但更保守 |
| Occupied | 不可通行 |
| 对角线 | 两个相邻 cardinal 格均可通行，禁止切角 |
| Footprint | Occupied 按 `vehicle_radius_m` 膨胀 |
| 规划范围 | 起点和目标包围框加默认 `16 m` margin |
| 资源限制 | 默认目标最远 `256 m`、最多 `100000` 格 |
| 单帧工作 | 最多 256 次 D* 扩展、256 个安全停车候选检查 |
| 跨帧扩展上限 | 当前有限规划窗格数 ×20；达到后报告 `expansion_limit` |
| 不可达目标 | 车体外缘距原目标不超过 `1 m`；在 `1 m + vehicle_radius_m` 中确定性采样连续候选 |

Unknown 可以通行是启动探索所必需：车辆开始时除出生附近外没有地图。如果要求 Unknown
不可通行，第一条 `goto` 将无法离开初始区域。驶向 Unknown waypoint 时，导航线速度默认
缩放到 `0.4`；最终仍受实时安全门控。

Unknown 不能作为最终安全停车点。候选的车辆 footprint 加 `0.25 m` 硬净空、候选所在
规划格以及该格中心到连续候选的末段连接都必须合法。已确认候选的整个包络还必须是
Free；选中后保持稳定，并在每帧证据更新后重新验证。

精确目标已确认危险、但附近只有对已知障碍安全且 D* 可达的未确认候选时，任务保持
`active`，进入 `goal_mode=approaching_safe_stop`，在 Unknown 中受安全门控抵近并随每帧
证据重新选点。抵达未确认候选只会停车等待扫描，不会伪报完成；包络确认 Free 后才转为
`nearby_safe`。到达已确认候选使用 `status=reached`、`reason=nearby_safe_stop`。
不存在任何几何安全且可达的候选时才报告 `nearby_safe_goal_unavailable`。

地图 delta 可包含 Unknown → Free、Unknown → Occupied、Occupied → Free 等双向变化。
规划器保留 `g`、`rhs`、优先队列与起点移动的 key modifier，只更新受变化格及 footprint
影响的顶点。更换目标或起点走出有限规划窗才重置搜索。

规划按控制帧增量推进。`goto` 已受理但当前切片尚未完成时，任务保持 `active`，
`planning=true` 且车辆保持 `stop`；旧 `path` 可继续出现在遥测中供 UI 对比，但不会被
执行。地图变化通过 `update(map_delta=...)` 触发相同额度的增量重规划，可能需要多个
控制帧才完成。
手动接管、显式 pause、连接断开和非法命令会停车、清除本次 pending 规划并保留任务；
显式 `resume` 会从当前位姿和地图重新启动。碰撞、无路、安全故障或定位丢失会将当前任务
置为 `blocked` 且不跳过队列；条件恢复后可 `resume` 重试，或用 `cancel_all` 清空。

已验证身份且未过期的 peer vehicle state 作为瞬态 footprint exclusion 单独进入规划，
不会写入 `ObservedGrid`。只有同一规划请求在忽略 peer exclusion 后存在可行路径，才把
当前无路归因为瞬态 peer 阻塞并保持 `active`；静态起点、目标或路径阻塞仍终止任务。
该分类复用一个只读 OwnMap D* 可达性证据，不读取共享地图或真值。已归因且租约仍有效的
peer 阻塞会等待 overlay 更新；匿名 LiDAR 动态阻塞即使 OwnMap 仍有路，也只获得一次有界
restart grace。单帧匿名遮挡消失后可恢复；持续或闪烁的匿名遮挡不会重置原有无路进度，
第二次失败稳定报告 `no_path`。持续存在的 OwnMap 墙同样不会被无限等待掩盖。规划给出的
期望速度进入 LocalSafety 前还会做 4 秒最近接近预测：发生冲突时，`vehicle_id` 字典序
较大的车辆让行并有界制动，连续 3 个控制 tick 确认冲突解除后恢复。让行期间 peer state
缺失或超过 `0.35 s` TTL 时不会盲目恢复；若粗 tick 制动后已经落入 peer 安全包络，只
允许经过 LocalSafety 的低速分离运动退出包络。显式取消或切换模式仍可解除任务。

### 滚动时域协同 Goto

`goto`、`patrol` 和 `coverage` 不增加新的车队任务类型；后两者展开的每个子目标继续进入
同一个 `GotoController`。可选的 grouped Coverage 只在本地确定性选择连续子矩形，不增加
P2P schema 或 motion-intent 版本。控制器复用 OwnMap-only D* 路径作为空间候选，并在每个
控制 tick 上为未来约 `4 s` 构造 prioritized SIPP 时间表。无冲突时首步 departure 不被
推迟；每次只提交/执行约 `0.8 s`，随后根据新的 odometry、D* 路径和 peer intent 滚动重算。

motion-intent v4 的 trajectory 由相对 `enter_offset_s` / `leave_offset_s` 组成。接收端以本地
receipt time 重建绝对区间，不比较不同车辆的 monotonic clock；intent generation、plan
generation 和严格递增 sequence 防止重启、旧计划和乱序包回灌。plan generation 只在 cell
序列、任务或 goal-hold 语义改变时增长，滚动重发的相对 offset 不会独自制造新代次。
显式 generation 必须是正的 u64；同一 generation 不得更改上述签名，所有字段校验成功后
才原子更新本地 generation/签名/待发布 intent，失败后下一次合法 generation 可继续。
相邻 trajectory cell 必须不同且满足 `next.enter > previous.leave`，禁止零时长 teleport；
单 cell 原地 hold 合法。commit 只允许
`0..min(0.8 s, trajectory 最后 leave_offset_s)`，其后的轨迹仅是未提交候选。

内部 `ReservationTable` 分别保存：

- cell reservation：防止两个车体在重叠时间占据同一空间包络；
- edge reservation：连续插值圆形 footprint，覆盖对向交换和几何线段交叉；
- goal reservation：到达车辆在 fresh lease 持续续租期间占据终点。

时间区间同时按发送车辆声明的加减速/通信 margin 和本车相同 margin 扩张；空间冲突使用
两车半径、定位不确定度可用时的 peer footprint，以及 `0.3 m` 自动安全余量。未提交候选
先按冲突等待年龄排序，再比较继承 owner 和 `vehicle_id`；双方已声明 commit 时只比较
owner/vehicle ID，避免每台车都因一跳延迟而误判自己的本地年龄更大。本地
`submitted_seq` 不跨车辆比较，`task_age_ticks` 当前也只用于协议观测；要让“更早任务”
安全参与跨车全序，需要后续加入 Lamport task token。占路车辆沿 current/target request
chain 递归继承最高优先级，并尝试一个 D* 已确认可通行的邻格；找不到时回退为等待，让
上游下一轮改时或改路。

当动态 peer exclusion 使当前单条 D* 候选暂时无路、但 OwnMap-only 路线仍存在时，控制器
可在完全观测的 Free 区域内搜索一条最长 `4 m` 的确定性让行路径。该搜索只寻找离开原路线
安全包络的 passing place，允许先沿路线绕过墙角再侧移；它不会穿过 Unknown/Occupied，
也不会扩展成第二套全局规划器。选中的单条让行路径仍交给同一 SIPP 和 LocalSafety 执行。

motion-intent v4 还包含必需的 `vacate_request` 字段；无请求时为 `null`，否则精确包含
`vehicle_id`、blocker footprint 所在的 `cell`，以及从请求车 `current_cell` 开始的
`route_cells`。路线窗口由请求车 OwnMap-only D* 路径生成，长度限制为 `2..64`；连续格不能
重复，锚点旋转量化后相邻格 Chebyshev 步长最多为 2。它是对单个已知车辆的短期请求，不是
共享地图、集中任务分派或多候选路径协议。

只有当前单条 D* 路线被可信 peer footprint 暂时截断、OwnMap 路线仍完整且请求车优先级更高
时，才向停在该路线上的 Auto/Idle peer 发布请求。请求随有序路线进度向前裁剪；请求车越过
blocker 后撤回，U/L 形路线不会因到路线终点的欧氏距离变小而提前撤回。目标车辆只在没有
活动或排队 Mission、请求与 peer-state 同 generation 且都 fresh、请求 cell 与自身物理
footprint 一致时响应。它复用真实 Goto、同一 SIPP 和 LocalSafety 驶到路线安全包络外，再
返回保存的原位；发布的 motion target/trajectory 始终对应真实运动，也不产生 Mission
事件。定位 lost、碰撞、safety stop、pause 或断联都会停车并保留 session。

请求者明确发布同 generation 的 fresh `vacate_request:null` 后，响应车还要求连续 3 tick
的 fresh pose/trajectory 均已离开请求包络，才开始返回；fresh 路线更新或 pause/resume 会
重置该 debounce。消息缺失、TTL 过期、身份/generation 不一致或物理 cell 不一致时冻结并
保持 fail-closed，不能把 lease 消失解释为车辆已经离开。证据活性只使用 MapSync 本地
receipt time；peer payload 的墙钟 timestamp 仅用于同源防回退，不与模拟时钟比较。

没有定向请求的活动 peer 仍使用 trajectory/goal reservation 与本车剩余 D* 路线的
cell/edge 冲突归因 blocker，同时保留继承 owner；无法归因、peer 证据失鲜或动态无路仍
存在时保持 fail-closed。仅由车体膨胀包络与路线相交、但自身时序轨迹会离开的 peer 继续
交给 SIPP 等待，不新建让行 session。priority root 也不会为已经继承该 root 的下游 blocker
新建或继续 implicit 让行；检测到这一跳环时会在同一 tick 回到普通 reservation/SIPP 仲裁。

expected peer 存在时，sidecar 未 ready、任一 peer-state/intent 缺失、TTL 过期，或同一来源
两类 topic 的 state/intent generation 不一致，都使新短前缀 fail-closed。SIPP 在固定 D*
候选上保留每个 `(path index, safe interval)` 的最早可达状态；中间格的等待/转向必须完整
落在同一安全区间，edge 或终点冲突可选择后续区间并从可行前驱重排。第一版没有额外的
网络 propose/ACK/commit 往返：它依赖上一 tick 的全员 fresh intent、确定性优先级和短
commit 前缀收敛；并发首次提案仍保留下一格租约、同步物理碰撞仲裁、LiDAR 与 LocalSafety
作为最终保障。当前 SIPP 只给一条 D* 空间候选和一条有界 passing-place detour 排时，不
搜索多条空间路径，也不是 CBS/PBS、联合最优 MAPF 或完整 PIBT 回溯。

### 直线窄走廊租约

`GotoController` 还用独立、只读的 OwnMap D* 路径检查前方是否存在完全观测、直线且内宽
不超过约 `3 m` 的瓶颈。弯曲、分支、过宽、观测不完整或无法在有界规划额度内确认的通道
不声明 corridor，继续使用普通下一格和轨迹冲突协调。检测不会消费 peer evidence，也
不会修改正在执行的 D* 路径。

检测到走廊后，车辆通过 motion-intent v4 发布有向 entry/exit cell。重叠的反向部分描述
可匹配为同一资源；未提交的下一轮候选使用等待年龄，冲突的 live claim 则由继承 owner 和
`vehicle_id` 给出所有节点一致的全序。winner 先发布
tentative claim，只有所有可见竞争者回传同一 owner（或无竞争声明连续稳定）后才进入。
任一时刻至多一个 confirmed owner；已确认 owner 规划时可忽略明确让给它的走廊等待者，
但 LocalSafety、动态感知和物理碰撞仲裁始终保留。

“无竞争”必须由成员视图明确证明：P2P 已 ready，且受信 expected allowlist 中每辆车都提供
同 session、通过 generation/sequence 校验且未过期的 motion intent；fresh `corridor:null`
表示该 peer 不竞争当前资源。缺失 intent、断联或 sidecar 未 ready 时，入口外 tentative
claim 不得自确认。已 confirmed 或已在廊内的 owner 不因随后分区被撤权，以免停车占住
单车道；它继续清空并正常释放。没有 map-sync/已知 peer 的真正单车保留 3 tick 自确认。

在 winner 出口侧，反向等待者按距各自 entry 的纵向格数和 `vehicle_id` 确定唯一 front；
只有 front 提前执行一次可逆侧移，同侧 rear waiter 保持原地。候选位置必须同时增加相对
owner 当前至目标/速度外推 sweep 的连续距离，以及相对实际行驶轴线的侧向距离；front 的
当前连续位姿和剩余侧移段都达到两车半径、定位不确定度与 `0.3 m` 余量之和才算安全。

ACK 只完成分布式 owner 收敛。owner 可在等待 ACK 或 front 清空时继续规划并接近 entry，
但按当前速度和制动能力在车体跨 entry 前闸停；peer state 缺失时 fail-closed。原进场位姿
只保存一次。租约仍活跃时 waiter 冻结导航；取得 owner 后，只有整个 saved segment 与目标
仍在本侧 entry 外才可先于 admission gate 原路返回，任何触及/跨 entry 的段都受 gate
约束。若上一 owner 已离开租约但 live peer 的安全包络占据 saved segment，则丢弃该旧段，
从侧袋当前位姿交回本地 D* 重规划；无遮挡时仍沿原段回归。释放边界包含远端 cell 外侧面、
车体半径和 `0.3 m` 自动安全余量，避免车尾尚未清空时提前交接。

走廊租约仍是通用时窗之上的拓扑特例。当前严格一次只允许一辆车通过，因此长走廊和深
队列吞吐近似线性；暂不可入廊 owner 跳过和同向 directional batching/convoy 留作后续
优化。它不把当前的递归 priority propagation + 单邻格回溯扩大宣称为完整 PIBT/MAPF。

遥测 `navigation` 中可观察：

```json
{
  "algorithm": "d_star_lite",
  "goal_mode": "nearby_safe",
  "goal": {"x_m": 9.0, "y_m": 4.0},
  "requested_goal": {
    "frame_id": "anchor_map",
    "x_m": 9.0,
    "y_m": 4.0
  },
  "effective_goal": {
    "frame_id": "anchor_map",
    "x_m": 8.5,
    "y_m": 3.5
  },
  "approach_distance_m": 0.207,
  "planning": false,
  "path_revision": 4,
  "replan_count": 2,
  "current_waypoint": {"x_m": 8.5, "y_m": 3.5},
  "path": [{"x_m": 7.5, "y_m": 3.5}],
  "planner_stats": {
    "expansions": 240,
    "incremental_updates": 43,
    "replans": 12,
    "resets": 1,
    "key_modifier_cost": 3.0,
    "candidate_inspections": 24
  }
}
```

公开路径和目标使用米；整数 cell 只存在于规划器内部。`goal` 保留原始
`global_map` 请求，`requested_goal` 与 `effective_goal` 分别明确原请求和实际执行的
`anchor_map` 坐标。`goal_mode` 为 `exact`、`approaching_safe_stop` 或 `nearby_safe`。
`approach_distance_m` 是实际目标到原请求目标的中心距离减去车体半径（下限为零），即
车体外缘距离，单位为米；因此车体中心允许距原目标超过 `1 m`。为限制遥测大小，`path`
最多报告前 64 个点。`planner_stats.candidate_inspections` 是当前 `goto` 已检查的安全
停车候选累计数。

## Scan matching 与局部 SLAM 边界

当前实现是确定性的 bounded correlative scan matcher：

- 将当前有效回波端点投影到候选 SE(2) 位姿；
- 在有限平移/旋转窗口搜索与旧 Occupied 格的一致性；
- 低支持、低得分、离群或最佳/次佳结果过于接近时拒绝修正；
- 匹配旧地图后才把当前扫描写入，避免自匹配；
- 已接受的修正进入后续 odometry 状态。

它能抑制有足够稳定结构时的小范围里程计漂移，但没有回环检测、地点识别、位姿图、全局
优化或持久化。因此这里只称为“最小局部 SLAM 前端”，不能等同于生产级 SLAM。定位进入
`lost` 后自动导航停车，扫描既不修正位姿也不写地图。

## A* 全真值调试工具

`a_star_search()` 和 CLI `pathfind` 仍保留，用于在生成的完整模拟地图上验证静态算法或
生成参考结果。它不是自动 `goto` 的数据源，也不会看到有限视野 Unknown 语义。

```bash
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200 --verbose
```

输出 waypoint 使用 `x_m` / `y_m`。库级 `a_star_search()` 的 tuple 是内部离散 cell：

```python
from mockvehicle2d.pathfinding import a_star_search

cell_path = a_star_search(
    grid,
    start=(0, 0),
    goal=(100, 100),
    vehicle_radius=0.5,
    resolution_m=1.0,
)
```

## 已知限制

- 局部地图和 D* Lite 状态只在进程内存中；重连保留，进程重启丢失。
- 控制连接断开会停车并暂停活动 `goto`；重连后需显式 `resume`。
- 暂无路径平滑、运动学轨迹优化和动态目标速度预测。
- prioritized SIPP 只调度一条 D* 空间候选和一个邻格 detour；尚无多候选时空搜索、完整
  PIBT backtracking、CBS/PBS oracle 或显式网络 propose/ACK/commit。
- 窄走廊只识别已完全观测的直线 `<=3 m` 类别，且严格单车通行；未实现 ready-owner
  skipping 或同向批处理。
- Unknown 的无回波 Free 更新沿用当前模拟约定，接入真实 Tmini 前必须校准。
- 水平 Tmini 无法发现落差；模拟器使用独立下视安全输入。
- 暂无回环、全局优化和中央地图同步。
