"""LLM client for natural language instruction parsing.

LLMClient — async client for llama.cpp server (OpenAI-compatible API)
"""

from __future__ import annotations

import json
import re


def _strip_thinking(content: str) -> str:
    """Strip <think>...</think> tags from LLM output.

    Handles both closed and unclosed (truncated) think blocks.
    After stripping, extracts remaining text for JSON parsing.
    """
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
    return content.strip()


_SYSTEM_PROMPT = """你是一个车辆指令解析器。将用户的自然语言指令转换为 JSON 格式。

当用户输入包含连接词（"然后"、"接着"、"再"、"之后"、";"、"并"、"并且"）时，表示多个连续指令。此时输出 JSON 数组 [{...}, {...}, ...]，每个元素是一个指令对象。
当用户输入是单个指令（无连接词）时，输出单个 JSON 对象。

每个指令对象包含两个字段：
- intent: 意图类型 (stop/goto/clarify/patrol)
- parameters: 与 intent 对应的参数对象

意图定义：
- stop: 用户要求立即停车。parameters 为 {}
- goto: 用户提供了明确的目标坐标 x_m 和 y_m（范围 0–255）。
  支持多种坐标格式："(100, 200)"、"100, 200"、"x=50 y=80" 等。
  注意：只有相对移动描述（如"前进3米"、"往左走"）而没有绝对坐标的，归为 clarify。
- patrol: 用户要求开始自动巡逻。parameters 为 {}
- clarify: 指令模糊、缺少关键参数、或无法匹配以上意图时使用。
  例如：缺少坐标的 goto、缺少角度的旋转、无意义输入、闲聊。

单指令示例：
"停" → {"intent": "stop", "parameters": {}}
"去坐标 (100, 200)" → {"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}}
"开到 10, 20" → {"intent": "goto", "parameters": {"x_m": 10, "y_m": 20}}
"前进 3 米" → {"intent": "clarify", "parameters": {"question": "请提供目标坐标", "missing_parameters": ["x_m", "y_m"]}}
"开到那边去" → {"intent": "clarify", "parameters": {"question": "请提供目标坐标", "missing_parameters": ["x_m", "y_m"]}}
"开始巡逻" → {"intent": "patrol", "parameters": {}}

多指令（含连接词）示例：
"去（200，100）巡逻" → [{"intent": "goto", "parameters": {"x_m": 200, "y_m": 100}}, {"intent": "patrol", "parameters": {}}]
"去 (10, 20) 然后去 (30, 40)" → [{"intent": "goto", "parameters": {"x_m": 10, "y_m": 20}}, {"intent": "goto", "parameters": {"x_m": 30, "y_m": 40}}]
"巡逻，然后去 (50, 50)" → [{"intent": "patrol", "parameters": {}}, {"intent": "goto", "parameters": {"x_m": 50, "y_m": 50}}]
"去 (5, 5) 接着停" → [{"intent": "goto", "parameters": {"x_m": 5, "y_m": 5}}, {"intent": "stop", "parameters": {}}]
"去 (100, 200)；巡逻" → [{"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}}, {"intent": "patrol", "parameters": {}}]
"去 (10, 10) 然后去 (20, 20) 再去 (30, 30)" → [{"intent": "goto", "parameters": {"x_m": 10, "y_m": 10}}, {"intent": "goto", "parameters": {"x_m": 20, "y_m": 20}}, {"intent": "goto", "parameters": {"x_m": 30, "y_m": 30}}]
"去 (100, 100) 然后巡逻然后停" → [{"intent": "goto", "parameters": {"x_m": 100, "y_m": 100}}, {"intent": "patrol", "parameters": {}}, {"intent": "stop", "parameters": {}}]

只输出 JSON，不要输出任何其他内容。"""


class LLMClient:
    """Async client for llama.cpp LLM inference (OpenAI-compatible API).

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API endpoint (default: llama.cpp local server).
    model : str
        Model name as registered in the server.
    max_retries : int
        Maximum number of retry attempts on JSON parse or schema validation failure.
    schema_validator : optional
        SchemaValidator instance for self-validation during retries.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen3-8B-Q4_K_M",
        max_retries: int = 3,
        schema_validator=None,
        enable_thinking: bool = True,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._max_retries = max_retries
        self._schema_validator = schema_validator
        self._enable_thinking = enable_thinking
        self._client = None  # Lazy init

    @property
    def _async_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self._base_url, api_key="not-needed")
        return self._client

    async def parse(self, text: str) -> list[dict]:
        """Send text to the LLM and return a list of parsed JSON instruction dicts.

        Retries on JSON decode errors and schema validation failures
        by appending error feedback to the conversation.

        Always returns a list:
        - Single instruction → len-1 list
        - Multi-instruction (with connectors) → list of dicts
        - Parse failure → empty list
        """
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                    extra_body={"enable_thinking": self._enable_thinking},
                    timeout=30.0,
                )
                content = response.choices[0].message.content
                if content is None:
                    return []

                # Strip <think>...</think> tags (even unclosed ones)
                content = _strip_thinking(content)
                # Strip markdown code fences if present
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    if attempt < self._max_retries:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"你的回复不是合法的 JSON。错误: {e}。请只输出 JSON。",
                        })
                        continue
                    return []

                # Normalize: dict → [dict], list → as-is
                if isinstance(result, dict):
                    instructions: list[dict] = [result]
                elif isinstance(result, list):
                    instructions = result
                else:
                    if attempt < self._max_retries:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": "你的回复必须是 JSON 对象或 JSON 数组。请重新输出。",
                        })
                        continue
                    return []

                # Schema validation: validate each element
                if self._schema_validator is not None:
                    all_valid = True
                    error_messages: list[str] = []
                    for i, inst in enumerate(instructions):
                        valid, error = self._schema_validator.validate(inst)
                        if not valid:
                            all_valid = False
                            error_messages.append(f"[{i}]: {error}")
                    if not all_valid:
                        if attempt < self._max_retries:
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": f"你的 JSON 不符合 schema。错误: {'; '.join(error_messages)}。请修正后重新输出。",
                            })
                            continue
                        return []

                return instructions

            except Exception:
                # Non-JSONDecodeError (e.g. timeout, connection error) — do NOT retry
                return []

        return []
