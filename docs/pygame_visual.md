# Mock Server — Pygame 可视化方案

## 概述

在现有碰撞检测模块基础上，用 Pygame 实现一个轻量 2D 可视化窗口，用于：
- 直观查看地图和车辆状态
- 手动驾驶车辆，实时验证碰撞检测
- 调试和演示

```
┌─────────────────────────────────────────────────────┐
│  mock_visual.py (pygame)                            │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │             地图渲染区                        │    │
│  │   · · · · █ · · ·                           │    │
│  │   · · · █ █ █ · ·      █ = 墙 (灰色)        │    │
│  │   · · · · █ · · ·      · = 空地 (浅色)       │    │
│  │   · · · · · · ● →     ● = 车辆 (蓝色圆)      │    │
│  │   · · · · · · · ·     → = 朝向指示           │    │
│  │   · · · · · · · ·     🔴 = 碰撞 (红色闪烁)    │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ 状态栏: Pose(5.3,2.7) Yaw:45° Collision:NO  │    │
│  │ [W/S/A/D] 移动 [Q/E] 转向 [R] 重置 [ESC] 退出 │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## 依赖

- `pygame` — `pip install pygame`

## 文件结构

```
test_tool/
├── mock_map_grid.py       ← 地图数据 (已有)
├── mock_collision.py      ← 碰撞检测 (已有)
├── mock_vehicle.py        ← WebSocket server (已有)
├── test_collision.py      ← 测试 (已有)
└── mock_visual.py         ← 🆕 Pygame 可视化入口
```

## 模块设计

### mock_visual.py

```
main()
  ├── pygame.init()
  ├── 加载地图: MapGrid.from_voxels() 或随机生成
  ├── 初始化车辆: VehicleState(x, y, yaw)
  │
  └── 主循环 (60fps):
      ├── 处理事件 (键盘/退出)
      ├── 更新车辆 (移动 + 碰撞检测)
      ├── 渲染地图 (Tile 网格)
      ├── 渲染车辆 (圆 + 方向线)
      └── 渲染 UI (状态栏)
```

### 车辆控制

| 按键 | 动作 | 说明 |
|------|------|------|
| W / ↑ | 前进 | 沿 yaw 方向移动 0.2m |
| S / ↓ | 后退 | 反方向移动 0.2m |
| A | 左旋 | yaw += 5° |
| D | 右旋 | yaw -= 5° |
| Q | 左平移 | 垂直于 yaw 向左移动 |
| E | 右平移 | 垂直于 yaw 向右移动 |
| R | 重置 | 回到起点 |
| ESC | 退出 | — |

### 碰撞反馈

```
正常: 蓝色圆 + 白色朝向线
碰撞预警: 圆变橙色（下一帧会撞）
碰撞: 圆变红色，车辆不移动
```

## 渲染细节

### 地图渲染

```
cell_size_px: 可配置 (默认 20px)
地图区域: grid.width × grid.height × cell_size_px

每帧绘制:
  for cell in visible_area:
      if wall:  fill_rect(灰色 #666)
      else:    fill_rect(浅灰 #DDD)
      网格线:  draw_rect(白色边框)
```

### 车辆渲染

```
圆心: world_to_screen(cx, cy)
半径: R * cell_size_px = 0.5 * 20 = 10px
方向: 从圆心沿 yaw 画一条长 1.5R 的线段

颜色:
  - 正常: 蓝色 (0, 128, 255)
  - 碰撞预警: 橙色 (255, 160, 0)
  - 碰撞: 红色 (255, 50, 50)
```

### 坐标转换

```python
def world_to_screen(wx: float, wy: float) -> tuple[int, int]:
    """世界坐标 (米) → 屏幕坐标 (px)"""
    screen_x = int(wx * CELL_SIZE_PX)
    screen_y = int(wy * CELL_SIZE_PX)
    return (screen_x, screen_y)
```

### 视口控制

```
初始: 显示完整地图，自动缩放到窗口大小
操作: 鼠标滚轮缩放，鼠标拖动平移
      (可选，先做固定视口)
```

## 碰撞检测集成

```python
# 每帧更新
def update_vehicle(keys, dt):
    # 1. 根据按键计算目标 pose
    new_x, new_y, new_yaw = compute_target(vehicle, keys, dt)

    # 2. 碰撞检测
    if is_circle_passable(grid, new_x, new_y, R):
        vehicle.x, vehicle.y, vehicle.yaw = new_x, new_y, new_yaw
        vehicle.colliding = False
    else:
        vehicle.colliding = True  # 不移动，变红
```

## 与现有代码的关系

```
mock_visual.py
  ├── import mock_map_grid     → MapGrid 类
  ├── import mock_collision    → is_circle_passable()
  └── 独立运行，不依赖 WebSocket
```

- **不影响** `mock_vehicle.py` (WebSocket server)
- **不影响** `test_collision.py` (测试)
- 共享 `mock_map_grid.py` 和 `mock_collision.py`
- 纯本地可视化工具，与 Pictor Godot 项目无关

## 实施步骤

- [ ] 1. 安装 pygame
- [ ] 2. 创建 `test_tool/mock_visual.py`
  - [ ] 2.1 MapGrid 地图加载 + 随机生成
  - [ ] 2.2 地图网格渲染
  - [ ] 2.3 车辆渲染 (圆 + 方向线)
  - [ ] 2.4 键盘控制 + 碰撞检测联动
  - [ ] 2.5 状态栏 UI
  - [ ] 2.6 碰撞颜色反馈
- [ ] 3. 手动测试：驾驶车辆撞墙，验证碰撞检测
- [ ] 4. 更新 `Task/task_11_mock_server.md`
