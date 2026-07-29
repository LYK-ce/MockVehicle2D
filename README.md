# MockVehicle2D

2D 车辆模拟器，Python 实现。用于 Pictor 项目的 WebSocket 协议测试、碰撞检测验证、
以及 **Qwen3-8B/14B 驱动的自然语言车辆指令系统**（NL→JSON→执行），
支持 stop / goto / clarify / patrol 四种意图。

## 快速开始

```bash
# 一键安装（创建 .venv + 安装依赖）
bash bootstrap.sh

# 然后激活环境
source .venv/bin/activate

# 运行测试
python -m pytest

# 启动可控 WebSocket Mock Server（默认端口 19090）
mockvehicle2d serve --vehicle-id mock_vehicle_01

# 端口被占用时改用其他端口，并在 Pictor 中输入相同端口
mockvehicle2d serve --port 9090 --vehicle-id mock_vehicle_01

# 校准模拟车（公开接口统一使用 SI：m、s、rad）
mockvehicle2d serve --linear-speed-mps 0.5 --angular-speed-rps 1.5708 \
  --vehicle-radius-m 0.5 --command-timeout-s 1.0

# 配置出生锚点与可重放的里程计误差
mockvehicle2d serve --anchor-id car_01_anchor --anchor-x-m 10 --anchor-y-m 10 \
  --anchor-yaw-rad 0 --odom-translation-noise-m 0.01 \
  --odom-yaw-noise-rad 0.0035 --odom-seed 42

# A* 全真值调试工具（不属于 goto 运行链路）
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200

# 自然语言指令（需要本地 llama.cpp server）
# 方式一：通过 CLI 一键启动 server
mockvehicle2d serve-llm                         # 默认 GPU 0, Qwen3-8B
mockvehicle2d serve-llm --gpu 0 --model Qwen3-14B-Q4_K_M  # 14B 模型

# 然后执行 NL 指令
mockvehicle2d nl "去坐标 (100, 200)"
mockvehicle2d nl "停"
mockvehicle2d nl "开始巡逻"
mockvehicle2d nl --interactive
mockvehicle2d nl --think "去坐标 (50, 30)"       # 开启 thinking 模式
mockvehicle2d nl --model Qwen3-14B-Q4_K_M "去坐标 (50, 30)"

# 离线评测
mockvehicle2d nl --eval
mockvehicle2d nl --eval --think                 # thinking 模式评测

# 启动 Pygame 可视化
mockvehicle2d visual

```

> **注意**：依赖安装在项目本地的 `.venv/` 中，不会污染系统 Python。

## 文件结构

```
MockVehicle2D/
├── src/mockvehicle2d/
│   ├── cli/                ← 统一 CLI 入口 (argparse)
│   │   └── main.py
│   ├── instruction/        ← NL→JSON 自然语言指令系统
│   │   ├── llm_client.py   ← LLMClient (Qwen via llama.cpp)
│   │   ├── dispatcher.py   ← v3 intent → function_call + Robot Controller 命令翻译
│   │   ├── validator.py    ← Schema 与可选地图语义校验
│   │   ├── state_machine.py← 指令生命周期与多指令队列
│   │   ├── compiler.py     ← 离线 JSON 指令编译/调试
│   │   ├── authority.py    ← 5 级权限仲裁
│   │   └── schemas/
│   │       └── v3.json     ← JSON Schema v3（stop/goto/clarify/patrol）
│   ├── pathfinding/        ← D* Lite 动态规划 + A* 全真值调试 + 路径跟随
│   │   ├── a_star.py
│   │   ├── d_star_lite.py
│   │   ├── waypoint_follower.py
│   │   └── __init__.py
│   ├── map_grid.py         ← MapGrid 类，2D 栅格地图 (bytearray, O(1))
│   ├── collision.py        ← 碰撞检测：Bresenham 线段 + AABB vs Circle
│   ├── vehicle.py          ← Server/Pygame 共用的运动、碰撞与指令看门狗
│   ├── local_state.py      ← 出生锚点、增量里程计与车辆自有观测地图
│   ├── navigation.py       ← 有限视野 D* Lite 重规划、局部目标跟踪与状态
│   ├── safety.py           ← Tmini/边缘观测、固定阈值与本地安全运行时
│   ├── server.py           ← WebSocket Server，接收 cmd/drive/goto/nl_command
│   ├── scan.py             ← YDLidar Tmini 二维角度/距离/强度扫描
│   └── visual.py           ← Pygame 可视化，支持 W+D 等组合驾驶与实时碰撞反馈
├── tests/
│   ├── test_collision.py   ← 碰撞检测测试套件
│   ├── test_pathfinding.py ← A* 寻路 + 路径跟随测试
│   ├── test_scan.py        ← 二维扫描几何测试
│   ├── test_vehicle.py     ← 指令、运动、看门狗和防穿墙测试
│   ├── test_goto.py        ← goto 协议、状态、接管和碰撞测试
│   ├── test_local_state.py ← 锚点、里程计、观测地图与定位质量测试
│   ├── test_scan_matching.py ← 有限窗配准、拒绝门控与漂移修正测试
│   ├── test_d_star_lite.py ← 增量规划、动态变化和参考最短路对照
│   ├── test_dynamic_navigation.py ← 有限视野发现、重规划与到达 E2E
│   ├── test_safety.py      ← 纯安全感知与策略测试
│   ├── test_safety_runtime.py ← 自动/手动安全运行时接入测试
│   ├── test_server_scan.py ← scan WebSocket 帧测试
│   ├── test_instruction.py ← NL 校验/状态机/编译器单元测试
│   ├── test_nl_integration.py ← NL 全链路集成测试
│   ├── nl_eval.json        ← 46 条评测数据集（stop/goto/clarify/patrol）
│   └── test_pathfinding_controller.py
├── docs/
│   ├── instruction/
│   │   └── PLAN.md         ← NL 系统设计文档
│   ├── mock_server.md
│   ├── pathfinding.md
│   ├── pygame_visual.md
│   └── websocket_protocol.md
├── pyproject.toml
├── LICENSE
└── README.md
```

## 车辆参数

| 参数 | 值 |
|------|------|
| 形状 | 圆形 |
| 半径 | 0.5 (直径 = 1 cell = 1m) |
| 航向角 | yaw (弧度), 0 = +x |

## 寻路

`goto` 只读取锚定局部位姿和 Tmini 累积出的有限视野 `ObservedGrid`。未知格可通行但
代价高于已确认自由格，且驶向未知格时降速；占用格按车辆半径膨胀后不可通行。每次扫描使
格子在 Unknown / Free / Occupied 间变化时，D* Lite 复用旧搜索状态增量修复路径。规划
按控制帧推进：默认每帧最多 256 次 D* 扩展和 256 个安全停车候选检查；未完成时
`navigation.planning=true`，车辆保持停止。

```bash
mockvehicle2d pathfind --start-m 10,10 --goal-m 200,200
```

| 组件 | 文件 | 说明 |
|------|------|------|
| D* Lite | `pathfinding/d_star_lite.py` | 有限规划窗、八连通、Unknown 高代价、半径膨胀、增量重规划 |
| 自动导航 | `navigation.py` | 将局部地图路径编译为速度，未知区降速并服从安全门控 |
| A* 调试 | `pathfinding/a_star.py` | 只供 `pathfind` 在完整模拟真值地图上做离线参考 |
| Legacy 跟随器 | `pathfinding/waypoint_follower.py` | 兼容旧的预编译静态路径调用 |

算法详情参见 [寻路文档](docs/pathfinding.md)。

## 碰撞检测

```
MapGrid (bytearray)
  cells[y * w + x]: 0 = 可通行, 1 = 墙, 2 = 无地面/落差
  操作: O(1)

Bresenham 线段碰撞
  逐格采样，任一格子不可通行（墙、落差或越界）→ 碰撞
  复杂度: O(max(dx, dy))

AABB vs Circle 圆形碰撞
  最近点 = (clamp(cx, gx, gx+1), clamp(cy, gy, gy+1))
  重叠 ⇔ 距离² < r²
```

## 测试

在源码仓库中运行 `python -m pytest` 执行 `tests/` 下的完整
测试集，包括有限视野 D* Lite、动态重规划、scan matching、SI 契约、WebSocket
协议、车辆运动、碰撞与安全回归；任一测试失败时命令返回非零状态。

## 自然语言指令系统（NL→JSON）

通过 Qwen 大模型将自然语言指令转换为结构化 JSON，再编译为确定性车辆任务执行。

### 架构

```text
用户 NL 输入 ("去坐标 (100, 200)")
    │
    ▼
┌──────────────────────────┐
│ LLM Client               │
│ LLMClient (Qwen 8B/14B) │  ← LLM 模式，通过 llama.cpp server
└──────────┬───────────────┘
           │
           ▼
    {"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}}
           │
           ▼
┌──────────────────────────┐
│ 三层校验                   │
│ 1. Schema (JSON Schema)   │  ← intent 枚举 / parameters 类型 / 数值范围
│ 2. Semantic (地图边界)      │  ← 坐标可通行性 / 距离上限
│ 3. Safety (SafetyRuntime) │  ← 净空 / 急停状态
└──────────┬───────────────┘
           │ 通过
           ▼
┌──────────────────────────┐
│ WebSocket 执行层           │
│ goto → 锚定局部状态        │
│      → 有限视野 D* Lite    │
│ stop  → 立即停车           │
│ patrol→ 启动巡逻           │
└──────────┬───────────────┘
           │
           ▼
       车辆执行
```

### 支持的意图 (4 种)

| 意图 | 中文示例 | JSON 参数 |
|------|---------|----------|
| `stop` | "停下"、"紧急停止" | `{}` |
| `goto` | "去 (100, 200)"、"开到 10, 20" | `{"x_m": 100, "y_m": 200}` |
| `clarify` | "开到那边去" → 反问坐标 | `{"question": "请指定坐标", "missing_parameters": [...]}` |
| `patrol` | "开始巡逻"、"启动巡逻" | `{}` |

### 运行模式

`mockvehicle2d nl` 通过 `LLMClient` 连接本地 llama.cpp server（`localhost:8000`）进行 NL→JSON 推理。默认单 GPU（`CUDA_VISIBLE_DEVICES=0`），Qwen3-8B-Q4_K_M，Thinking 模式默认关闭。

### 翻译层

服务端在 LLM 输出 v3 意图 JSON 后，通过 `dispatcher.py` 确定性翻译为：
- **function_call** — 内部 dispatch 格式（如 `{"name":"goto","arguments":{"x_m":100,"y_m":200}}`）
- **command** — Robot Controller 协议命令（如 `{"cmd":"auto","action":"push","missions":[{"type":"goto","x_m":100,"y_m":200}]}`）

翻译产物在所有 NL 相关 WebSocket 回复（`nl_parse_result`、`nl_task_update`、`nl_confirm_request`）中以 `function_call` 和 `command` 字段携带。Pictor 可直接消费 `command` 字段对接已有的 mode/manual/auto 命令处理。

### LLM 配置

```bash
# 默认 Qwen3-8B (8-bit 量化)
mockvehicle2d nl "去坐标 (100, 200)"

# 切换到 Qwen3-14B
mockvehicle2d nl --model Qwen3-14B-Q4_K_M "去坐标 (100, 200)"

# 开启 / 关闭 thinking 模式（默认 off）
mockvehicle2d nl --think "去坐标 (100, 200)"
mockvehicle2d nl --no-think "去坐标 (100, 200)"
```

### JSON Schema v3（最小化设计）

LLM 仅需输出 2 个字段：

```json
{"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}}
```

移除了 `schema_version`、`timestamp`、`confidence`、`reasoning`（均不被下游消费）。
`additionalProperties: true` 确保 LLM 偶发的额外字段不会阻塞指令。

### Retry 机制

LLM 输出 JSON 解析失败或 Schema 校验失败时，自动将错误反馈给 LLM 进行最多 3 次重试。
超时和连接错误不重试。

### 离线评测

```bash
mockvehicle2d nl --eval
```

54 条测试覆盖全部 4 种意图 + 边界情况（越界坐标、注入攻击、乱码输入）。
Qwen3-8B: 94.4% 意图准确率，100% Schema 通过率（单 GPU A100, Q4_K_M，thinking 开/关结果相同）。

## 通信协议

遵循 [WebSocket 通信协议](docs/websocket_protocol.md)。

启动 Server 后，在 Pictor 中连接 `ws://127.0.0.1:19090`；使用 `--port 9090` 时，
Pictor 也应连接 `ws://127.0.0.1:9090`。连接首帧固定为
`hello` 除 `vehicle_id` 外还包含 `map` 元数据（分辨率、尺寸、二进制 chunk 布局及
`simulator_map → global_map` 变换），随后依次发送 `map_full → pose → scan`。

> 旧版 Pictor 若忽略 `hello.map`，只适用于默认出生锚点产生的恒等变换。使用非默认
> anchor 时，消费者必须应用 `transform_to_global_map`，否则地图叠加位置必然错误。

| 方向 | 消息 | 状态 |
|------|------|------|
| 上行 (Server→Pictor) | `hello` | ✅ |
| 上行 (Server→Pictor) | `map_full` | ✅ |
| 上行 | `pose` | ✅ |
| 上行 | `scan` | ✅ |
| 上行 | `map_delta` | ⏸️ |
| 上行 (Server→Pictor) | `nl_parse_result` / `nl_confirm_request` / `nl_task_update` / `nl_scan_report` | ✅ |
| 下行 (Pictor→Server) | `cmd` / `drive` / `goto` | ✅ |
| 下行 (Pictor→Server) | `nl_command` / `nl_clarify_response` | ✅ |

`scan` 默认使用 Tmini 轮廓：360°、0.02–12 m、名义 4000 Hz 测距、名义 6 Hz 扫描、667 条均匀射线。有效回波按 0.01 m 量化；无回波为 `range: 0.0, intensity: 0.0`，不能当作障碍物。水平 Tmini 只返回墙体；确定性的 `state=2` 落差测试区会显示在 `map_full`，由模拟的向下地面探测输入负责安全判断。

每辆车启动时只知道 `global_map → anchor_map` 的出生锚点。随后用运动增量生成
`anchor_map → odom → base_link → lidar` 的锚定里程计，并把 Tmini 扫描累计到独立的
`ObservedGrid`（`Unknown/Free/Occupied/Forbidden`）。其中下视安全输入发现的落差写为
不会被水平雷达 Free 射线覆盖的 `Forbidden`。该地图及 revision 在 Pictor 断开重连后仍保留，
但当前只存在小车内存中，不上传中央地图。每帧先用 bounded correlative scan matching
将当前扫描与旧的 Occupied 证据配准，通过支持数、得分和歧义 margin 后才修正里程计，再
写入当前帧。默认噪声为零以保持确定性；非零噪声使用 `--odom-seed` 重放。这是最小局部
SLAM 前端，不含回环检测、位姿图或全局优化，不能宣称生产级定位。

控制器可发送离散命令 `{"type":"cmd","seq":1,"cmd":"forward"}`，也可发送连续速度 `{"type":"drive","seq":2,"linear_mps":0.25,"angular_rps":-0.4}`；Server 都立即返回 `cmd_ack`。`drive` 的绝对值上限分别由 `--linear-speed-mps` 和 `--angular-speed-rps` 配置。超过 `--command-timeout-s` 未收到有效非零命令、收到非法命令或连接断开时，车辆自动停止；碰撞时停在最后一个安全位置。旧 `cmd` 格式保持兼容并与 `drive` 使用同一运动、碰撞和看门狗逻辑。

发送 `{"type":"goto","seq":3,"x_m":12.0,"y_m":8.5}` 可让模拟车自主前往
`global_map` 中的目标；Server 在锚点边界将其转换为 `anchor_map` 坐标。`pose` 顶层坐标仍
兼容 Pictor，但来源为 `anchored_odometry`，并附带局部位姿协方差、quality 和 revision。
控制器使用有限规划窗内的 D* Lite 路径；尚未观测的障碍只有进入 Tmini 视野、写入局部图后
才触发增量绕行。定位 `degraded` 时自动线速限制为一半；定位 `lost` 时自动任务停车并变为
`blocked`，且停止写入本地观测地图。自动
行驶在障碍/边缘净空 `0.25–1.0 m` 内线性降速，净空 `<=0.25 m` 或安全输入故障时停车；
任何手动 `cmd`/`drive` 或非法输入也会取消活动目标。
`goto_ack.accepted=true` 只表示目标进入 `active`；长距离或重规划任务可能仍在
`planning=true`，此时遥测可保留旧路径用于显示，但不会执行，车辆命令为 `stop`。地图
delta 触发的重规划同样分帧推进。手动接管、碰撞、安全阻断或定位丢失都会清除未完成规划；
终止后必须重新发送 `goto`，调用重规划不会恢复旧任务。
若已观测证据确认精确目标无法容纳车体和 `0.25 m` 硬净空，或 D* Lite 确认精确目标
不可达，控制器会在“车体外缘距原目标不超过 `1 m`”的范围内确定性采样连续候选；因此
车体中心允许位于 `1 m + vehicle_radius_m` 范围内。已确认 Free 的安全候选可直接作为
`nearby_safe` 目标；只有未确认但对已知障碍安全且可达的候选时，车辆以
`approaching_safe_stop` 受安全门控抵近并随扫描重新验证，抵达未确认点不会伪报完成。
不存在任何安全可达候选时才 `blocked`。遥测中的 `goal` 始终保留原请求，
`requested_goal`、`effective_goal` 和 `approach_distance_m` 分别给出 `anchor_map`
原目标、实际目标和以米表示的车体外缘距离。

`pose.safety` 持续报告 `{state, reason, obstacle_clearance_m, edge_clearance_m,
edge_point_vehicle_m}`。障碍净空按圆形车体沿行驶方向扫过的完整走廊计算，不使用固定角度扇区；运行时把延迟时段拆成不超过 `0.05 m` 且不越过硬停止净空的小步，每步重新观测。手动驾驶不在慢速区降速，但仍执行硬停止和故障停车；新的安全方向命令可解除手动安全锁停，纯旋转允许用于脱困。Tmini 只负责正障碍距离，落差净空及车辆坐标系证据点是模拟的辅助下视/相机输入，不能解释为雷达能力。

`hello.map.source` 将二进制 `map_full` 标为 `simulator_ground_truth`，仅用于物理碰撞、生成传感器数据和
调试显示；正常 WebSocket `pose` 不再泄露绝对真值。模拟 scan 生成器仍必须读取真值环境，
但导航只消费锚定里程计、车辆自有地图和局部安全结果。无回波目前按最大量程更新为自由空间，这是当前
模拟协议约定，接入真实 Tmini 前必须按硬件无回波语义校准。水平 Tmini 不能检测跌落。
当前已实现有限视野局部栅格、轻量 scan matching 和 D* Lite；仍没有回环、全局优化、地图
持久化或中央地图同步。
