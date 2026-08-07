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
D* Lite（增量修复路径）──► 局部 waypoint ──► 期望速度
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

### 直线窄走廊租约

`GotoController` 还用独立、只读的 OwnMap D* 路径检查前方是否存在完全观测、直线且内宽
不超过约 `3 m` 的瓶颈。弯曲、分支、过宽、观测不完整或无法在有界规划额度内确认的通道
不声明 corridor，继续使用普通下一格和轨迹冲突协调。检测不会消费 peer evidence，也
不会修改正在执行的 D* 路径。

检测到走廊后，车辆通过 motion-intent v2 发布有向 entry/exit cell。重叠的反向部分描述
可匹配为同一资源；等待年龄决定优先级，`vehicle_id` 只作最终 tie-break。winner 先发布
tentative claim，只有所有可见竞争者回传同一 owner（或无竞争声明连续稳定）后才进入。
任一时刻至多一个 confirmed owner；已确认 owner 规划时可忽略明确让给它的走廊等待者，
但 LocalSafety、动态感知和物理碰撞仲裁始终保留。

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

这套机制是 PIBT 启发的去中心化短租约，不是完整 PIBT/MAPF。当前严格一次只允许一辆车
通过，因此长走廊和深队列吞吐近似线性；暂不可入廊 owner 跳过、同向 directional
batching/convoy 和 SIPP 时间窗均留作后续优化。

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
- 窄走廊只识别已完全观测的直线 `<=3 m` 类别，且严格单车通行；未实现 ready-owner
  skipping、同向批处理或 SIPP。
- Unknown 的无回波 Free 更新沿用当前模拟约定，接入真实 Tmini 前必须校准。
- 水平 Tmini 无法发现落差；模拟器使用独立下视安全输入。
- 暂无回环、全局优化和中央地图同步。
