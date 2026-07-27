"""LLM clients for natural language instruction parsing.

FakeModelClient — rule-based deterministic parser for offline testing
LLMClient       — async client for llama.cpp server (OpenAI-compatible API)
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


class FakeModelClient:
    """Rule-based deterministic parser for offline testing.

    Supports basic Chinese patterns for all seven intents.
    """

    def parse(self, text: str) -> dict | None:
        """Parse NL text into a structured instruction dict, or None on failure."""
        text = text.strip()
        if not text:
            return self._clarify("输入为空，请提供指令", [])

        result = self._try_parse(text)
        if result is not None:
            return result
        return self._clarify(f"无法理解指令「{text}」，请使用坐标指定目标位置", [])

    def _try_parse(self, text: str) -> dict | None:
        # stop
        if re.match(r"^(停|停下|停止|紧急停止|别动了)$", text):
            return self._make_instruction("stop", {})

        # status
        if re.match(r"^(现在什么状态|到哪了|有没有问题|状态|在哪)$", text):
            return self._make_instruction("status", {})

        # goto_point
        m = self._parse_goto_point(text)
        if m:
            return self._make_instruction("goto_point", {"x_m": m[0], "y_m": m[1]})

        # move_distance
        m = self._parse_move_distance(text)
        if m:
            return self._make_instruction("move_distance", m)

        # rotate
        m = self._parse_rotate(text)
        if m:
            return self._make_instruction("rotate", m)

        # scan_report
        m = self._parse_scan(text)
        if m is not None:
            return self._make_instruction("scan_report", m)

        return None

    # ── pattern parsers ──────────────────────────────────────

    @staticmethod
    def _parse_goto_point(text: str) -> tuple[float, float] | None:
        # "去 (x, y)" / "去坐标 (x, y)" / "开到 x, y" / "前往 (x, y)"
        patterns = [
            r"^去\s*\(\s*(-?[\d.]+)\s*[,，]\s*(-?[\d.]+)\s*\)$",
            r"^去坐标\s*\(\s*(-?[\d.]+)\s*[,，]\s*(-?[\d.]+)\s*\)$",
            r"^开到\s*(-?[\d.]+)\s*[,，]\s*(-?[\d.]+)$",
            r"^前往\s*\(\s*(-?[\d.]+)\s*[,，]\s*(-?[\d.]+)\s*\)$",
        ]
        for pat in patterns:
            m = re.match(pat, text)
            if m:
                try:
                    return float(m.group(1)), float(m.group(2))
                except ValueError:
                    return None
        return None

    @staticmethod
    def _parse_move_distance(text: str) -> dict | None:
        # "前进 N 米" / "后退 N 米"
        m = re.match(r"^前进\s*([\d.]+)\s*米$", text)
        if m:
            return {"distance_m": float(m.group(1)), "direction": "forward"}
        m = re.match(r"^后退\s*([\d.]+)\s*米$", text)
        if m:
            return {"distance_m": float(m.group(1)), "direction": "backward"}
        return None

    @staticmethod
    def _parse_rotate(text: str) -> dict | None:
        # "左转 N 度" / "右转 N 度"
        m = re.match(r"^左转\s*([\d.]+)\s*度$", text)
        if m:
            return {"angle_deg": float(m.group(1)), "direction": "left"}
        m = re.match(r"^右转\s*([\d.]+)\s*度$", text)
        if m:
            return {"angle_deg": float(m.group(1)), "direction": "right"}
        return None

    @staticmethod
    def _parse_scan(text: str) -> dict | None:
        if text in ("看一下", "扫一圈", "扫描一下", "扫描"):
            return {}
        m = re.match(r"^(前面|左边|右边|后面|周围)(有什么|有障碍吗|有东西吗)$", text)
        if m:
            query_map = {"前面": "前方", "左边": "左侧", "右边": "右侧", "后面": "后方", "周围": "四周"}
            return {"query": query_map.get(m.group(1), "")}
        return None

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _make_instruction(intent: str, params: dict) -> dict:
        return {"intent": intent, "parameters": params}

    @staticmethod
    def _clarify(question: str, missing: list[str]) -> dict:
        return {
            "intent": "clarify",
            "parameters": {
                "question": question,
                "missing_parameters": missing,
            },
        }


_SYSTEM_PROMPT = """你是一个车辆指令解析器。将用户的自然语言指令转换为 JSON 格式。

只输出 JSON 对象，包含两个字段：
- intent: 意图类型 (stop/status/goto_point/move_distance/rotate/scan_report/clarify)
- parameters: 与 intent 对应的参数对象

意图与参数：
- stop: 无参数，parameters 为 {}
- status: 无参数，parameters 为 {}
- goto_point: x_m (数字) 和 y_m (数字)
- move_distance: distance_m (数字, 0.01-10.0) 和 direction ("forward" 或 "backward")
- rotate: angle_deg (数字, -360到360) 和 direction ("left" 或 "right")
- scan_report: 可选 query (字符串)
- clarify: question (字符串)，可选 missing_parameters (字符串数组)

示例：
"停" → {"intent": "stop", "parameters": {}}
"状态" → {"intent": "status", "parameters": {}}
"去坐标 (100, 200)" → {"intent": "goto_point", "parameters": {"x_m": 100, "y_m": 200}}
"前进 3 米" → {"intent": "move_distance", "parameters": {"distance_m": 3.0, "direction": "forward"}}
"左转 90 度" → {"intent": "rotate", "parameters": {"angle_deg": 90, "direction": "left"}}
"前面有什么" → {"intent": "scan_report", "parameters": {"query": "前方"}}

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
