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
    # Remove closed think blocks
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    # If an unclosed <think> remains (truncated output), remove it and everything after
    content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
    return content.strip()


_SYSTEM_PROMPT = """你是一个车辆指令解析器。将用户的自然语言指令转换为 JSON 格式。

只输出 JSON 对象，包含两个字段：
- intent: 意图类型 (stop/goto/clarify/patrol)
- parameters: 与 intent 对应的参数对象

意图与参数：
- stop: 无参数，parameters 为 {}
- goto: x_m (数字) 和 y_m (数字)
- patrol: 无参数，parameters 为 {}
- clarify: question (字符串)，可选 missing_parameters (字符串数组)

示例：
"停" → {"intent": "stop", "parameters": {}}
"去坐标 (100, 200)" → {"intent": "goto", "parameters": {"x_m": 100, "y_m": 200}}
"开到那边去" → {"intent": "clarify", "parameters": {"question": "请提供目标坐标", "missing_parameters": ["x_m", "y_m"]}}
"开始巡逻" → {"intent": "patrol", "parameters": {}}

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
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._max_retries = max_retries
        self._schema_validator = schema_validator
        self._client = None  # Lazy init

    @property
    def _async_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self._base_url, api_key="not-needed")
        return self._client

    async def parse(self, text: str) -> dict | None:
        """Send text to the LLM and return parsed JSON dict.

        Retries on JSON decode errors and schema validation failures
        by appending error feedback to the conversation.

        Returns None on timeout or after exhausting retries.
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
                    extra_body={"enable_thinking": True},
                    timeout=30.0,
                )
                content = response.choices[0].message.content
                if content is None:
                    return None

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
                    return None

                # Schema validation retry
                if self._schema_validator is not None:
                    valid, error = self._schema_validator.validate(result)
                    if not valid:
                        if attempt < self._max_retries:
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": f"你的 JSON 不符合 schema。错误: {error}。请修正后重新输出。",
                            })
                            continue
                        return None

                return result

            except Exception:
                # Non-JSONDecodeError (e.g. timeout, connection error) — do NOT retry
                return None

        return None
