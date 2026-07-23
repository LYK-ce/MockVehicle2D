# Pygame 本地可视化

`mockvehicle2d visual` 启动一个独立的 24×24 栅格仿真窗口，用于快速检查
车辆运动和碰撞逻辑。它与 WebSocket Server 不共享运行状态；联调 Pictor 时应运行
`mockvehicle2d serve`。

## 操作

| 按键 | 动作 |
|------|------|
| W / ↑ | 前进 |
| S / ↓ | 后退 |
| A / ← | 左转 |
| D / → | 右转 |
| W+A、W+D、S+A、S+D | 平移并转向 |
| R | 重置 |
| Esc | 退出 |

按键使用实时按住状态：按住持续运动，松开即停止。车辆正常时显示为蓝色，碰撞
截停后显示为红色；底部状态栏显示位姿、当前命令和碰撞状态。

## 范围

- 与 WebSocket Server 共用 `MapGrid`、`Vehicle` 和连续防穿墙检测。
- 使用固定 25 px/cell、0.5 m 车体半径和 60 FPS。
- 不显示 Tmini 扫描、`goto`、安全净空或 Pictor 的局部视野效果。
