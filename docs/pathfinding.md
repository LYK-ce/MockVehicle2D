# 寻路

车辆可从任意起点自动规划避障路径并导航到终点。由两个组件组成：A* 搜索和 WaypointFollower。

## 架构

```
MapGrid                    Vehicle
  │                          │
  ▼                          ▼
a_star_search()  ──────►  WaypointFollower
  │                          │
  ▼                          ▼
list[(x,y)] 路径        (cmd, done) 每帧
```

- **A\*** 在 MapGrid 上计算最短路径（静态地图，不感知动态障碍）
- **WaypointFollower** 将网格路径转换为 Vehicle cmd 序列，逐帧跟踪

## A* 搜索

### 算法参数

| 参数 | 值 |
|------|------|
| 连通性 | 八连通（含对角线） |
| 对角线代价 | √2 |
| Cardinal 代价 | 1.0 |
| 启发式 | 欧几里得距离（八连通 admissible） |
| 对角线剪枝 | 需两个相邻 cardinal 格均可通行 |

### 车辆半径适配

车辆半径 r=0.5 意味着当车辆中心在某个 cell 内时，其圆形区域会覆盖该 cell 的 8 邻域。因此对墙做 **1-cell 膨胀**：所有墙 cell 及其 8 个邻居均标记为不可通行。

```
r=0 时:         r=0.5 时 (膨胀后):
. . . . .       # # # . .
. W . . .       # W # . .
. . . . .       # # # . .
. = 可通行      F = . 可通行
W = 墙          # = 膨胀封锁区
```

`vehicle_radius=0` 可关闭膨胀（如测试无半径车辆时）。

### API

```python
from mockvehicle2d.pathfinding import a_star_search

path = a_star_search(grid, start=(0, 0), goal=(100, 100), vehicle_radius=0.5)
# → [(0,0), (1,1), ..., (100,100)] 或 None（无路径）
```

### CLI

```bash
mockvehicle2d pathfind --start 10,10 --goal 200,200
# Path found: 266 waypoints

mockvehicle2d pathfind --start 10,10 --goal 200,200 --verbose
# [0] (10, 10)
# [1] (11, 11)
# ...
```

## 路径跟随器

### 状态机

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
  ┌───────────┐   |δ| < 10°   ┌──────────┐     │
  │ turn to   │ ────────────► │ forward  │     │
  │ target    │               │          │     │
  └───────────┘               └────┬─────┘     │
       ▲                           │           │
       │     |δ| ≥ 10°             │           │
       └───────────────────────────┘           │
                                               │
                   距离 < 0.5m                  │
               ┌────────────────────────────────┘
               ▼
         ┌──────────┐
         │ arrived  │  → (stop, True)
         └──────────┘
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `arrival_distance` | 0.5 m | 距终点此距离内视为到达 |
| `waypoint_distance` | 0.5 m | 距当前途经点此距离内则推进到下一个 |
| `angle_tolerance` | 10° | 朝向偏差小于此值则直行，否则转弯 |

### API

```python
from mockvehicle2d.pathfinding import WaypointFollower

follower = WaypointFollower(path)

# 每物理帧调用：
cmd, done = follower.next_cmd(vehicle.x, vehicle.y, vehicle.yaw)
# cmd ∈ {"forward", "spin_left", "spin_right", "stop"}
# done = True 表示到达终点
```

## 限制

- 路径在**静态地图**上计算，不感知动态障碍
- 无路径平滑 / 后处理——路径是网格坐标序列
- 跟随器不处理碰撞——Vehicle 自身的防穿墙机制仍生效
- 不感知 FOV / 未知区域
