# WebSocket 通信协议

## 基本信息

| 项目 | 值 |
|------|------|
| 传输协议 | WebSocket |
| 数据格式 | JSON 文本消息 |
| 编码 | UTF-8 |
| 角色 | 小车 = Server，PC = Client |
| 默认端口 | 9090 |

每条消息为单行 JSON，顶层必有 `type` 字段。

坐标系：2D 用 `(x, y)`，3D 高度用 `z`，与 Godot 坐标系统一。

---

## 上行：小车 → PC

### pose — 车辆位姿

实时发送车辆位置、朝向和速度。

```json
{
    "type": "pose",
    "ts": 1717800000.123,
    "x": 1.5,
    "y": 3.2,
    "z": 0.0,
    "yaw": 0.785,
    "vx": 0.5,
    "vy": 0.0
}
```

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `ts` | f64 | 秒 | Unix 时间戳 |
| `x`, `y` | f32 | 米 | 2D 世界坐标 |
| `z` | f32 | 米 | 高度 |
| `yaw` | f32 | 弧度 | 偏航角 |
| `vx`, `vy` | f32 | 米/秒 | 2D 速度分量 |

---

### scan — 本地二维激光扫描

每次 `pose` 后发送一帧 YDLidar 风格的局部扫描。它只表示小车相对坐标系中的障碍物距离，**不**提供定位、点云、SLAM 或全局地图。

```json
{
    "type": "scan",
    "ts": 1717800000.124,
    "frame_id": "laser",
    "config": {
        "min_angle": -3.141593,
        "max_angle": 3.124139,
        "angle_increment": 0.017453,
        "time_increment": 0.000278,
        "scan_time": 0.1,
        "min_range": 0.05,
        "max_range": 12.0,
        "point_count": 360,
        "angle_unit": "rad",
        "range_unit": "m",
        "angle_direction": "clockwise_from_forward",
        "no_return": {"range": null, "intensity": 0.0}
    },
    "points": [
        {"angle": 0.0, "range": 2.5, "intensity": 1.0},
        {"angle": 0.017453, "range": null, "intensity": 0.0}
    ]
}
```

`angle` 的单位是弧度，`range` 的单位是米，字段形式与 YDLidar SDK 的 `LaserPoint` 对齐。`yaw=0`、`angle=0` 指向世界 `+x`；在本模拟器的 `+y` 向下栅格中，正角度朝 `+y` 增长（从上方看顺时针）。

无回波统一编码为 `range: null, intensity: 0.0`，包括最大量程外、离开已知栅格或未知空间；它**不是**距离为 `max_range` 的障碍物。首个墙体命中使用其栅格边界距离，强度固定为 `1.0`。扫描是确定性射线投射，没有噪声或反射率模型。

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | string | 固定为小车本地 `laser` 坐标系 |
| `config` | object | 一帧的 LaserScan 配置与无回波约定 |
| `points` | array | 按角度递增排列的 `{angle, range, intensity}` 读数 |

---

### map_full — 全量地图

连接建立或重连后发送完整地图。

```json
{
    "type": "map_full",
    "ts": 1717800000.200,
    "voxels": [
        {"gx": 0, "gy": 0, "gz": 0, "state": 0, "conf": 0.95},
        {"gx": 1, "gy": 0, "gz": 0, "state": 1, "conf": 0.80}
    ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts` | f64 | Unix 时间戳 |
| `voxels` | array | 全量体素列表 |
| `gx`, `gy` | i32 | 2D 网格坐标 |
| `gz` | i32 | 高度层 |
| `state` | u8 | 0=可通行 1=不可通行 |
| `conf` | f32 | 置信度 0.0~1.0 |

---

### map_delta — 增量地图

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

```json
{
    "cmd": "forward"
}
```

| 命令 | 说明 |
|------|------|
| `forward` | 前进 |
| `backward` | 后退 |
| `spin_left` | 左旋 |
| `spin_right` | 右旋 |
| `stop` | 停止（松手时发送） |

---

## 消息一览

```
上行 (小车 → PC)          下行 (PC → 小车)
─────────────────         ─────────────────
pose                       cmd
scan
map_full
map_delta
```
