# NL→指令系统：v3 JSON → 函数调用翻译层

> 创建日期：2026-07-29
> 状态：设计中（待审查）

---

## 1. 概述

### 1.1 核心思路

**LLM 完全不动**。在服务端新增一个翻译层（`dispatcher.py`），将 LLM 输出的 v3 JSON 翻译为：

1. **Python 函数调用**（内部 dispatch）
2. **Robot Controller 协议命令**（WebSocket `command` 字段，给 Pictor）

```
LLM 输出（不变）       翻译层（新增）           执行层（不变）
─────────────────    ──────────────────    ──────────────────
{"intent":"goto",    → func_call:          → GotoController.start()
 "parameters":         {"name":"goto",
  {"x_m":100,          "arguments":
   "y_m":200}}         {"x_m":100,"y_m":200}}

                       ↓
                      command:
                      {"cmd":"auto",
                       "action":"push",
                       "missions":[
                         {"type":"goto",
                          "x_m":100,"y_m":200}]}
```

### 1.2 为什么不在 LLM 侧做 function calling

| 方案 | 优点 | 缺点 |
|------|------|------|
| LLM 直接输出 function call | GBNF grammar 约束，JSON 必定合法 | 改 LLM prompt、改 client、需要调试新输出格式 |
| **服务端翻译（本方案）** | **LLM 零改动、现有评测全部保留、零风险** | 翻译层是确定性代码（这不是缺点） |

翻译层是**纯确定性逻辑**（查表 + 字段重命名），不涉及概率模型。比改 LLM prompt 可靠得多。

### 1.3 包含的内容

- 新增：`dispatcher.py` — v3 intent → function call → robot controller command 翻译
- 修改：`server.py` — 用翻译产物替换硬编码 `if/elif`
- 修改：`docs/websocket_protocol.md` — 新增 `function_call` 和 `command` 字段

---

## 2. 架构总览

```
                           ┌──────────────────────────┐
                           │  LLMClient.parse()        │
                           │  (完全不变)                │
                           │  输出: list[dict] v3 JSON  │
                           └──────────┬───────────────┘
                                      │
                        [{"intent":"goto",
                          "parameters":{"x_m":100,"y_m":200}},
                         {"intent":"stop",
                          "parameters":{}}]
                                      │
                                      ▼
                           ┌──────────────────────────┐
                           │  translate()  [新增]       │
                           │                           │
                           │  v3 intent → function_call │
                           │      ↓                    │
                           │  function_call → command   │
                           │  (Robot Controller 协议)    │
                           └──────────┬───────────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                        ▼                           ▼
              ┌─────────────────┐         ┌─────────────────┐
              │  function_call  │         │    command      │
              │ (内部 dispatch)  │         │ (WebSocket 回复) │
              │                 │         │                 │
              │ {"name":"goto", │         │ {"cmd":"auto",  │
              │  "arguments":   │         │  "action":"push",│
              │  {"x_m":100,    │         │  "missions":     │
              │   "y_m":200}}   │         │  [{"type":"goto",│
              │                 │         │    "x_m":100,    │
              │     │           │         │    "y_m":200}]}  │
              └─────┼───────────┘         └────────┬────────┘
                    │                              │
                    ▼                              ▼
              ┌──────────┐              ┌──────────────────┐
              │ dispatch │              │ nl_parse_result   │
              │ 查表调用  │              │ + function_call   │
              │ Vehicle  │              │ + command         │
              │ API      │              │ + instruction     │
              └──────────┘              │ (三字段并行)       │
                                       └──────────────────┘
```

**LLM 输出和 Vehicle API 完全没变。只新增了中间翻译层和 WebSocket 输出字段。**

---

## 3. 翻译表：v3 intent → function_call

### 3.1 翻译规则

```python
_V3_TO_FUNCTION_CALL = {
    "stop": lambda params: {
        "name": "stop",
        "arguments": {},
        "command": {"cmd": "manual", "action": "stop"},
    },
    "goto": lambda params: {
        "name": "goto",
        "arguments": {"x_m": params["x_m"], "y_m": params["y_m"]},
        "command": {
            "cmd": "auto",
            "action": "push",
            "missions": [{"type": "goto", "x_m": params["x_m"], "y_m": params["y_m"]}],
        },
    },
    "patrol": lambda params: {
        "name": "patrol",
        "arguments": {},
        "command": {
            "cmd": "auto",
            "action": "push",
            "missions": [{"type": "patrol"}],
        },
    },
    "clarify": lambda params: {
        "name": "clarify",
        "arguments": {
            "question": params.get("question", "请提供更多信息"),
            "missing_parameters": params.get("missing_parameters", []),
        },
        "command": None,  # clarify 不需要 robot controller 命令
    },
}
```

**输入 LLM 的 v3 JSON，输出三个产物：**

| 产物 | 用途 | 消费者 |
|------|------|--------|
| `function_call` | 内部 dispatch → Vehicle API | `server.py` |
| `command` | Robot Controller 协议 | WebSocket → Pictor |
| `instruction` | v3 原始 JSON（不变） | 向后兼容 |

### 3.2 dispatch 表：function_call → Vehicle API

```python
def dispatch(function_call: dict, vehicle, navigation, local_state):
    """查表调用底层 Vehicle API。"""
    name = function_call["name"]
    args = function_call["arguments"]

    if name == "stop":
        vehicle.stop()
        if navigation:
            navigation.cancel("nl_stop")

    elif name == "goto":
        # 和现在 server.py 中 _execute_parsed_instruction 完全一样的逻辑
        vehicle.stop()
        _start_estimated_goto(navigation, vehicle, local_state, args["x_m"], args["y_m"])

    elif name == "patrol":
        # 和现在完全一样（空桩）
        pass

    elif name == "clarify":
        # 不走 Vehicle API，走状态机 CONFIRMING
        raise ClarifyRequest(args["question"], args.get("missing_parameters", []))
```

**这就是当前 `server.py:_execute_parsed_instruction()` 里的 `if/elif` 分支，只是从 server.py 搬到了 dispatcher.py。**

---

## 4. 新增文件：`dispatcher.py`

### 4.1 完整接口

```python
# src/mockvehicle2d/instruction/dispatcher.py

"""将 v3 NL 意图 JSON 翻译为函数调用和 Robot Controller 协议命令。"""

from dataclasses import dataclass
from typing import Any


class ClarifyRequest(Exception):
    """clarify 意图：不走 Vehicle API，触发状态机 CONFIRMING。"""
    def __init__(self, question: str, missing_parameters: list[str]):
        self.question = question
        self.missing_parameters = missing_parameters


@dataclass
class TranslatedInstruction:
    """单条指令的翻译产物。"""
    function_call: dict[str, Any]       # {"name": "goto", "arguments": {...}}
    command: dict[str, Any] | None      # Robot Controller 协议，clarify 时为 None
    instruction: dict[str, Any]         # 原始 v3 JSON（向后兼容）


def translate(instruction: dict) -> TranslatedInstruction:
    """v3 intent JSON → function_call + command + instruction。

    Parameters
    ----------
    instruction : dict
        单条 v3 指令，如 {"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}}

    Returns
    -------
    TranslatedInstruction

    Raises
    ------
    ValueError
        intent 不在翻译表中。
    """
    intent = instruction.get("intent")
    params = instruction.get("parameters", {}) or {}

    translator = _TRANSLATORS.get(intent)
    if translator is None:
        raise ValueError(f"unknown intent: {intent}")

    fc = translator(params)
    return TranslatedInstruction(
        function_call={"name": fc["name"], "arguments": fc["arguments"]},
        command=fc["command"],
        instruction=instruction,
    )


def translate_all(instructions: list[dict]) -> list[TranslatedInstruction]:
    """批量翻译。"""
    return [translate(inst) for inst in instructions]


# ── 翻译表 ────────────────────────────────────────────

def _translate_stop(params: dict) -> dict:
    return {
        "name": "stop",
        "arguments": {},
        "command": {"cmd": "manual", "action": "stop"},
    }


def _translate_goto(params: dict) -> dict:
    x = params["x_m"]
    y = params["y_m"]
    return {
        "name": "goto",
        "arguments": {"x_m": x, "y_m": y},
        "command": {
            "cmd": "auto",
            "action": "push",
            "missions": [{"type": "goto", "x_m": x, "y_m": y}],
        },
    }


def _translate_patrol(params: dict) -> dict:
    return {
        "name": "patrol",
        "arguments": {},
        "command": {
            "cmd": "auto",
            "action": "push",
            "missions": [{"type": "patrol"}],
        },
    }


def _translate_clarify(params: dict) -> dict:
    return {
        "name": "clarify",
        "arguments": {
            "question": params.get("question", "请提供更多信息"),
            "missing_parameters": params.get("missing_parameters", []),
        },
        "command": None,
    }


_TRANSLATORS = {
    "stop":    _translate_stop,
    "goto":    _translate_goto,
    "patrol":  _translate_patrol,
    "clarify": _translate_clarify,
}
```

### 4.2 行数估计

整个 `dispatcher.py` 约 **120 行**。其中翻译表 60 行，`translate()` 20 行，`dispatch()` 30 行，docstring 10 行。

---

## 5. `server.py` 改动

### 5.1 当前：硬编码 `if/elif`

```python
# server.py _execute_parsed_instruction() 中（简化）
if intent == "stop":
    vehicle.stop()
    navigation.cancel("nl_stop")
elif intent == "patrol":
    # stub
    state_machine.transition(InstructionState.ACTIVE)
elif intent == "goto":
    vehicle.stop()
    _start_estimated_goto(navigation, vehicle, local_state, x_m, y_m)
elif intent == "clarify":
    state_machine.transition(InstructionState.CONFIRMING)
    ...
```

### 5.2 改后：用翻译层

```python
# server.py 中

from mockvehicle2d.instruction.dispatcher import (
    translate_all, ClarifyRequest, TranslatedInstruction,
)

async def _handle_nl_command(...):
    instructions = await nl_client.parse(text)
    if not instructions:
        ...  # 解析失败，和现在一样

    # [新增] 翻译
    try:
        translated = translate_all(instructions)
    except ValueError as e:
        ...  # unknown intent

    state_machine.enqueue(instructions)  # 不变
    instruction = state_machine.dequeue_next()

    # [新增] 取出当前指令对应的翻译产物
    ti = translated[state_machine.current_index - 1]

    result = await _execute_parsed_instruction(
        ..., translated=ti  # [新增参数]
    )

    # [新增] 回复中注入 function_call 和 command
    result["function_call"] = ti.function_call
    result["command"] = ti.command
    # result["instruction"] 已经存在，不变

    return result


async def _execute_parsed_instruction(..., translated: TranslatedInstruction):
    """现在用 function_call dispatch 而不是 if/elif intent。"""

    fc = translated.function_call
    name = fc["name"]
    args = fc["arguments"]

    if name == "clarify":
        # clarify 特殊处理：不走 Vehicle API
        state_machine.transition(InstructionState.CONFIRMING)
        await send({
            "type": "nl_confirm_request",
            "question": args["question"],
            "missing_parameters": args.get("missing_parameters", []),
            "instruction": translated.instruction,
            "function_call": fc,
            "command": None,
        })
        return

    # 所有非 clarify 指令：走 dispatch
    _dispatch_to_vehicle(name, args, vehicle, navigation, local_state)
    state_machine.transition(InstructionState.ACTIVE)


def _dispatch_to_vehicle(name, args, vehicle, navigation, local_state):
    """查表调用 Vehicle API。"""
    if name == "stop":
        vehicle.stop()
        if navigation:
            navigation.cancel("nl_stop")

    elif name == "goto":
        vehicle.stop()
        _start_estimated_goto(
            navigation, vehicle, local_state,
            args["x_m"], args["y_m"],
        )

    elif name == "patrol":
        pass  # stub，和现在一样

    else:
        raise ValueError(f"unknown function: {name}")
```

### 5.3 改动摘要

| `server.py` 位置 | 改动 | 量 |
|---|---|---|
| `_handle_nl_command()` | 调用 `translate_all()`，注入新字段 | +8 行 |
| `_execute_parsed_instruction()` | 改为 `if name == "clarify"` + `_dispatch_to_vehicle()` | 重构 ~30 行 |
| 新增 `_dispatch_to_vehicle()` | 查表 dispatch，等价于原 `if/elif` 分支 | ~15 行 |

---

## 6. WebSocket 协议更新

### 6.1 `nl_parse_result` — 新增三个字段

```json
{
  "type": "nl_parse_result",
  "accepted": true,

  "result_type": "function_call",

  "function_call": {
    "name": "goto",
    "arguments": {"x_m": 100, "y_m": 200}
  },

  "command": {
    "cmd": "auto",
    "action": "push",
    "missions": [{"type": "goto", "x_m": 100, "y_m": 200}]
  },

  "instruction": {"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}},

  "sequence_index": 1,
  "sequence_total": 1
}
```

### 6.2 字段说明

| 字段 | 来源 | 作用 |
|------|------|------|
| `result_type` | 新增 | 始终为 `"function_call"`（所有 v3 intent 都会被翻译） |
| `function_call` | 新增 | `translate()` 产物。内部 dispatch 格式 |
| `command` | 新增 | `translate()` 产物。Robot Controller 协议，Pictor 直接消费 |
| `instruction` | 不变 | 原始 v3 JSON，向后兼容 |

### 6.3 Pictor 消费方式

Pictor 已实现 mode/manual/auto 三层协议（July 28 的 commits）。它只需读 `command` 字段即可：

```gdscript
# Pictor 中（示意）
if msg.get("command"):
    handle_robot_command(msg["command"])
    # → {"cmd": "auto", "action": "push", "missions": [...]}
    # → 已对接的分发逻辑
else:
    # 回退：读 instruction
    handle_instruction(msg["instruction"])
```

### 6.4 `nl_task_update` — 同样新增

```json
{
  "type": "nl_task_update",
  "status": "active",
  "function_call": {"name": "goto", "arguments": {"x_m": 100, "y_m": 200}},
  "command": {"cmd": "auto", "action": "push", "missions": [{"type": "goto", "x_m": 100, "y_m": 200}]},
  "instruction": {"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}},
  "sequence_index": 1,
  "sequence_total": 2
}
```

### 6.5 `nl_confirm_request` — 同样新增

```json
{
  "type": "nl_confirm_request",
  "question": "请指定目标坐标",
  "missing_parameters": ["x_m", "y_m"],
  "function_call": {"name": "clarify", "arguments": {"question": "请指定目标坐标", "missing_parameters": ["x_m", "y_m"]}},
  "command": null,
  "instruction": {"intent": "clarify", "parameters": {"question": "请指定目标坐标", "missing_parameters": ["x_m", "y_m"]}}
}
```

---

## 7. 文件变更清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/mockvehicle2d/instruction/dispatcher.py` | **新建** | `translate()`, `translate_all()`, `TranslatedInstruction`, `ClarifyRequest`, 翻译表 (~120 行) |
| `src/mockvehicle2d/instruction/__init__.py` | 修改 | 导出 `translate`, `translate_all`, `TranslatedInstruction`, `ClarifyRequest` |
| `src/mockvehicle2d/server.py` | 修改 | `_handle_nl_command()` 注入新字段；`_execute_parsed_instruction()` 用 function_call dispatch |
| `docs/websocket_protocol.md` | 修改 | 新增 `result_type`, `function_call`, `command` 字段文档 |
| `docs/nl_function_calling_design.md` | 保留 | 本文档 |
| `tests/test_instruction.py` | 追加 | `translate()` 单元测试 |

**LLM 相关文件完全不动：**
- `llm_client.py` — 不动
- `validator.py` — 不动
- `compiler.py` — 不动
- `schemas/v3.json` — 不动
- `state_machine.py` — 不动（instruction 仍然是 v3 JSON）
- `nl_eval.json` — 不动

---

## 8. 翻译覆盖

### 8.1 当前 v3 的 4 个 intent 全部覆盖

| v3 intent | → function_call | → command |
|-----------|----------------|-----------|
| `stop` | `{"name": "stop", "arguments": {}}` | `{"cmd": "manual", "action": "stop"}` |
| `goto` | `{"name": "goto", "arguments": {"x_m":..., "y_m":...}}` | `{"cmd": "auto", "action": "push", "missions": [...]}` |
| `patrol` | `{"name": "patrol", "arguments": {}}` | `{"cmd": "auto", "action": "push", "missions": [{"type": "patrol"}]}` |
| `clarify` | `{"name": "clarify", "arguments": {"question":..., ...}}` | `null` |

### 8.2 robot_controller.md 对照

当前 v3 只翻译了 robot_controller.md 三层指令的子集。其余 function（`switch_mode`, `move`, `spin`, `pause`, `resume`, `cancel`, `lidar_scan`）**已在翻译表结构和 dispatch 逻辑中预留**，等 LLM prompt 扩展支持这些意图后再启用。

**预留方式**：`_TRANSLATORS` 表只注册了 4 个 intent。未来 LLM prompt 扩展支持新 intent 时，只需在翻译表追加条目：

```python
# 未来：LLM 学会输出 {"intent": "move", "parameters": {"direction": "forward"}}
# 只需在 _TRANSLATORS 加一行：
_TRANSLATORS["move"] = _translate_move
```

---

## 9. 实施计划

只有一个 Phase，全部一起做。量很小。

| 步骤 | 内容 | 文件 |
|------|------|------|
| 1 | 新建 `dispatcher.py` | 1 个新文件，~120 行 |
| 2 | 修改 `server.py` | 重构 `_execute_parsed_instruction`，注入新字段 |
| 3 | 更新 `websocket_protocol.md` | 新增字段文档 |
| 4 | 运行现有的全部测试 | `pytest tests/ -v` — 确保 0 失败 |
| 5 | 追加 `translate()` 单元测试 | 覆盖 4 个 intent 的翻译正确性 |

预估总改动量：**~200 行新增 + ~30 行重构**。

---

> **下一步**：审查后可以开始实施。LLM 零改动，风险极低。
