# 有限视野寻路

自动 `goto` 使用车辆自己的有限视野占据栅格和 D* Lite。模拟器完整真值地图不参与路径
规划，只用于物理碰撞、传感器生成和调试显示。

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
D* Lite（增量修复路径）──► 局部 waypoint ──► 速度控制 ──► safety
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
| 不可达目标 | 在原目标 `1 m` 内选择安全包络均已确认 Free 且 D* 可达的最近 cell center |

Unknown 可以通行是启动探索所必需：车辆开始时除出生附近外没有地图。如果要求 Unknown
不可通行，第一条 `goto` 将无法离开初始区域。驶向 Unknown waypoint 时，导航线速度默认
缩放到 `0.4`；最终仍受实时安全门控。

Unknown 不能作为不可达目标的安全停车点。候选的车辆 footprint 加 `0.25 m` 硬净空所覆盖
的所有格都必须已经是 Free；候选按距原目标的距离和 cell 坐标确定性排序，再由同一个
D* Lite 验证从当前估计位姿可达。选中后执行目标保持稳定，除非它也变得危险或不可达。
到达候选仍使用终态 `reached`，但原因为 `nearby_safe_stop`，不会伪报
`goal_tolerance`。

地图 delta 可包含 Unknown → Free、Unknown → Occupied、Occupied → Free 等双向变化。
规划器保留 `g`、`rhs`、优先队列与起点移动的 key modifier，只更新受变化格及 footprint
影响的顶点。更换目标或起点走出有限规划窗才重置搜索。

遥测 `navigation` 中可观察：

```json
{
  "algorithm": "d_star_lite",
  "goal_mode": "nearby_safe",
  "effective_goal": {
    "frame_id": "anchor_map",
    "x_m": 8.5,
    "y_m": 3.5
  },
  "path_revision": 4,
  "replan_count": 2,
  "current_waypoint": {"x_m": 8.5, "y_m": 3.5},
  "path": [{"x_m": 7.5, "y_m": 3.5}],
  "planner_stats": {
    "expansions": 240,
    "incremental_updates": 43,
    "replans": 12,
    "resets": 1,
    "key_modifier_cost": 3.0
  }
}
```

公开路径和目标使用米；整数 cell 只存在于规划器内部。`goal` 保留原始 global-map 请求，
`effective_goal` 是实际执行的 `anchor_map` 目标，`goal_mode` 为 `exact` 或
`nearby_safe`。为限制遥测大小，`path` 最多报告前 64 个点。

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
- 控制连接断开会取消活动 `goto`，重连后需重新下发目标。
- 暂无路径平滑、运动学轨迹优化和动态目标速度预测。
- Unknown 的无回波 Free 更新沿用当前模拟约定，接入真实 Tmini 前必须校准。
- 水平 Tmini 无法发现落差；模拟器使用独立下视安全输入。
- 暂无回环、全局优化和中央地图同步。
