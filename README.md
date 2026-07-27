# MockVehicle2D

2D 车辆模拟器，Python 实现。用于 Pictor 项目的 WebSocket 协议测试、碰撞检测验证、
以及 **Qwen 驱动的自然语言车辆指令系统**（NL→JSON→执行）。

## 快速开始

```bash
# 一键安装（创建 .venv + 安装依赖）
bash bootstrap.sh

# 然后激活环境
source .venv/bin/activate

# 运行测试
mockvehicle2d test

# 启动可控 WebSocket Mock Server（默认端口 19090）
mockvehicle2d serve --vehicle-id mock_vehicle_01

# 端口被占用时改用其他端口，并在 Pictor 中输入相同端口
mockvehicle2d serve --port 9090 --vehicle-id mock_vehicle_01

# 校准模拟车（角速度单位为度/秒）
mockvehicle2d serve --linear-speed 0.5 --angular-speed 90 --vehicle-radius 0.5 --command-timeout 1.0

# A* 寻路（在 256×256 随机地图上规划路径）
mockvehicle2d pathfind --start 10,10 --goal 200,200

# 自然语言指令（需要本地 llama.cpp server）
# 方式一：通过 CLI 一键启动 server
mockvehicle2d serve-llm                         # 默认 GPU 0, Qwen3-8B
mockvehicle2d serve-llm --gpu 0 --model Qwen3-14B-Q4_K_M  # 14B 模型

# 方式二：手动启动 server（脚本）
bash scripts/start_llm_server.sh                # 默认
bash scripts/start_llm_server.sh 0 Qwen3-14B-Q4_K_M  # 14B

# 然后执行 NL 指令
mockvehicle2d nl "去坐标 (100, 200)"
mockvehicle2d nl "前进 3 米"
mockvehicle2d nl "左转 90 度"
mockvehicle2d nl --interactive
mockvehicle2d nl --model Qwen3-14B-Q4_K_M "前面有什么"

# 离线评测
mockvehicle2d nl --eval

# 启动 Pygame 可视化
mockvehicle2d visual

# 或通过 Python 模块运行
python -m mockvehicle2d test
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
│   │   ├── validator.py    ← 三层校验：Schema → 语义 → 安全
│   │   ├── state_machine.py← 11 状态指令生命周期
│   │   ├── compiler.py     ← JSON 指令 → 可执行任务
│   │   ├── authority.py    ← 5 级权限仲裁
│   │   └── schemas/
│   │       └── v2.json     ← JSON Schema v2（仅 intent + parameters）
│   ├── pathfinding/        ← A* 寻路 + 路径跟随
│   │   ├── a_star.py
│   │   ├── waypoint_follower.py
│   │   └── __init__.py
│   ├── map_grid.py         ← MapGrid 类，2D 栅格地图 (bytearray, O(1))
│   ├── collision.py        ← 碰撞检测：Bresenham 线段 + AABB vs Circle
│   ├── vehicle.py          ← Server/Pygame 共用的运动、碰撞与指令看门狗
│   ├── navigation.py       ← local odom 直达目标控制与状态
│   ├── safety.py           ← Tmini/边缘观测、固定阈值与本地安全运行时
│   ├── server.py           ← WebSocket Server，接收 cmd/drive/goto/nl_command
│   ├── scan.py             ← YDLidar Tmini 二维角度/距离/强度扫描
│   └── visual.py           ← Pygame 可视化，支持 W+D 等组合驾驶与实时碰撞反馈
├── tests/
│   ├── test_instruction.py ← NL 校验/状态机/编译器单元测试
│   ├── test_nl_integration.py ← NL 全链路集成测试
│   ├── nl_eval.json        ← 51 条评测数据集
│   ├── test_collision.py
│   ├── test_pathfinding.py
│   ├── test_pathfinding_controller.py
│   ├── test_scan.py
│   ├── test_vehicle.py
│   ├── test_goto.py
│   ├── test_safety.py
│   ├── test_safety_runtime.py
│   └── test_server_scan.py
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

车辆可从任意起点自动规划避障路径并导航到终点。

```bash
mockvehicle2d pathfind --start 10,10 --goal 200,200
```

| 组件 | 文件 | 说明 |
|------|------|------|
| A* 搜索 | `pathfinding/a_star.py` | 八连通，欧几里得启发式，对角线剪枝，1-cell 膨胀 |
| 路径跟随 | `pathfinding/waypoint_follower.py` | 网格路径 → Vehicle cmd 序列，朝向跟踪 |

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

`mockvehicle2d test` 会运行栅格/碰撞、Tmini 扫描、车辆运动、WebSocket
协议、`goto`、安全策略、延迟执行、NL 指令解析与集成回归测试（247 条）。

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
    {"intent": "goto_point", "parameters": {"x_m": 100, "y_m": 200}}
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
│ TaskCompiler              │
│ goto_point → GotoController│
│         或 A* + PathFollow│
│ rotate    → 航向计算       │
│ stop      → 立即停车       │
└──────────┬───────────────┘
           │
           ▼
       车辆执行
```

### 支持的意图 (7 种)

| 意图 | 中文示例 | JSON 参数 |
|------|---------|----------|
| `stop` | "停下"、"紧急停止" | `{}` |
| `status` | "现在什么状态"、"在哪" | `{}` |
| `goto_point` | "去 (100, 200)"、"开到 10, 20" | `{"x_m": 100, "y_m": 200}` |
| `move_distance` | "前进 3 米"、"后退 1.5 米" | `{"distance_m": 3.0, "direction": "forward"}` |
| `rotate` | "左转 90 度"、"右转 45 度" | `{"angle_deg": 90, "direction": "left"}` |
| `scan_report` | "前面有什么"、"扫一圈" | `{"query": "前方"}` |
| `clarify` | "开到那边去" → 反问坐标 | `{"question": "请指定坐标", "missing_parameters": [...]}` |

### 运行模式

`mockvehicle2d nl` 通过 `LLMClient` 连接本地 llama.cpp server（`localhost:8000`）进行 NL→JSON 推理。

### LLM 配置

```bash
# 默认 Qwen3-8B (8-bit 量化)
mockvehicle2d nl "去坐标 (100, 200)"

# 切换到 Qwen3-14B
mockvehicle2d nl --model Qwen3-14B-Q4_K_M "去坐标 (100, 200)"
```

### JSON Schema v2（最小化设计）

LLM 仅需输出 2 个字段——大幅降低小模型出错概率：

```json
{"intent": "goto_point", "parameters": {"x_m": 100, "y_m": 200}}
```

移除了 `schema_version`、`timestamp`、`confidence`、`reasoning`（均不被下游消费）。
`additionalProperties: true` 确保 LLM 偶发的额外字段不会阻塞指令。

### Retry 机制

LLM 输出 JSON 解析失败或 Schema 校验失败时，自动将错误反馈给 LLM 进行最多 3 次重试。
超时和连接错误不重试（infra 问题不是 LLM 质量问题）。

> **当前限制：单指令模式。** 一次 NL 输入仅处理一条指令（如 "去 (10,20)"）。
> 复合指令（如 "去 (10,20) 然后左转 90 度"）需要分两次发送，或未来扩展多指令序列支持。

### 离线评测

```bash
mockvehicle2d nl --eval
```

51 条测试覆盖全部 7 种意图 + 边界情况（越界坐标、注入攻击、乱码输入）。

## 通信协议

遵循 [WebSocket 通信协议](docs/websocket_protocol.md)。

启动 Server 后，在 Pictor 中连接 `ws://127.0.0.1:19090`；使用 `--port 9090` 时，
Pictor 也应连接 `ws://127.0.0.1:9090`。连接首帧固定为
`{"type":"hello","vehicle_id":"mock_vehicle_01"}`，随后依次发送 `map_full → pose → scan`。

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

控制器可发送离散命令 `{"type":"cmd","seq":1,"cmd":"forward"}`，也可发送连续速度 `{"type":"drive","seq":2,"linear_mps":0.25,"angular_rps":-0.4}`；Server 都立即返回 `cmd_ack`。`drive` 的绝对值上限分别由 `--linear-speed` 和 `--angular-speed` 配置。超过 `--command-timeout` 未收到有效非零命令、收到非法命令或连接断开时，车辆自动停止；碰撞时停在最后一个安全位置。旧 `cmd` 格式保持兼容并与 `drive` 使用同一运动、碰撞和看门狗逻辑。

发送 `{"type":"goto","seq":3,"x_m":12.0,"y_m":8.5}` 可让模拟车在本地 `odom` 坐标中直达目标，Server 返回 `goto_ack`，并在 `pose.control_mode` 与 `pose.navigation` 中持续报告模式、状态、目标和结束原因。当前定位输入是 `simulator_ground_truth`；控制器只会先转向、直线前进和接近目标减速，不会规划绕行。自动行驶在障碍/边缘净空 `0.25–1.0 m` 内线性降速，净空 `<=0.25 m` 或安全输入故障时停车并报告 `blocked`；停止后不会自行恢复。任何手动 `cmd`/`drive` 或非法输入也会取消活动目标。

`pose.safety` 持续报告 `{state, reason, obstacle_clearance_m, edge_clearance_m}`。障碍净空按圆形车体沿行驶方向扫过的完整走廊计算，不使用固定角度扇区；运行时把延迟时段拆成不超过 `0.05 m` 且不越过硬停止净空的小步，每步重新观测。手动驾驶不在慢速区降速，但仍执行硬停止和故障停车；新的安全方向命令可解除手动安全锁停，纯旋转允许用于脱困。Tmini 只负责正障碍距离，落差净空是模拟的辅助下视/相机输入，不能解释为雷达能力。

`map_full` 与 `pose` 标有 `source: "simulator_ground_truth"`，仅供仿真验收和可视化；只有 `scan` 是模拟的 Tmini 本地观测。未来导航算法不能把真值消息当作真实传感器输入。当前没有实现相机、定位误差、雷达噪声或 `map_delta`。
