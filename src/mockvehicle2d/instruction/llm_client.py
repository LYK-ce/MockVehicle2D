"""LLM clients for natural language instruction parsing.

FakeModelClient — rule-based deterministic parser for offline testing
VLLMClient      — async client for local vLLM (OpenAI-compatible API)
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone, timedelta

# Beijing timezone (UTC+8)
_BEIJING_TZ = timezone(timedelta(hours=8))


def _beijing_now() -> str:
    """Return current Beijing time as ISO 8601 string."""
    return datetime.now(_BEIJING_TZ).isoformat()


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
        return {
            "schema_version": "1.0",
            "intent": intent,
            "timestamp": _beijing_now(),
            "parameters": params,
            "confidence": 0.95,
            "reasoning": f"fake model: matched {intent} pattern",
        }

    @staticmethod
    def _clarify(question: str, missing: list[str]) -> dict:
        return {
            "schema_version": "1.0",
            "intent": "clarify",
            "timestamp": _beijing_now(),
            "parameters": {
                "question": question,
                "missing_parameters": missing,
            },
            "confidence": 0.6,
            "reasoning": "fake model: unable to match any pattern",
        }


_SYSTEM_PROMPT = """你是一个车辆指令解析器。将用户的自然语言指令转换为 JSON 格式。

你必须输出一个 JSON 对象，包含以下字段：
- schema_version: 固定为 "1.0"
- intent: 意图类型，取值为 stop, status, goto_point, move_distance, rotate, scan_report, clarify
- timestamp: 当前时间 ISO 8601 格式
- parameters: 与 intent 对应的参数对象
- confidence: 0.0-1.0 之间的置信度
- reasoning: 简短的推理说明（最多500字符）

意图与参数对应关系：
- stop: 无需参数，parameters 为空对象 {}
- status: 无需参数，parameters 为空对象 {}
- goto_point: 需要 x_m (数字) 和 y_m (数字)
- move_distance: 需要 distance_m (数字, 0.01-10.0) 和 direction ("forward" 或 "backward")
- rotate: 需要 angle_deg (数字, -360到360) 和 direction ("left" 或 "right")
- scan_report: 可选 query (字符串)
- clarify: 需要 question (字符串)，可选 missing_parameters (字符串数组)

示例：
输入: "去坐标 (100, 200)"
输出: {"schema_version": "1.0", "intent": "goto_point", "timestamp": "2026-01-01T00:00:00+08:00", "parameters": {"x_m": 100, "y_m": 200}, "confidence": 0.95, "reasoning": "用户指定了明确的目标坐标"}

输入: "停"
输出: {"schema_version": "1.0", "intent": "stop", "timestamp": "2026-01-01T00:00:00+08:00", "parameters": {}, "confidence": 0.99, "reasoning": "用户要求停止"}

输入: "前进 3 米"
输出: {"schema_version": "1.0", "intent": "move_distance", "timestamp": "2026-01-01T00:00:00+08:00", "parameters": {"distance_m": 3.0, "direction": "forward"}, "confidence": 0.95, "reasoning": "用户要求向前移动指定距离"}

输入: "左转 90 度"
输出: {"schema_version": "1.0", "intent": "rotate", "timestamp": "2026-01-01T00:00:00+08:00", "parameters": {"angle_deg": 90, "direction": "left"}, "confidence": 0.95, "reasoning": "用户要求左转指定角度"}

输入: "前面有什么"
输出: {"schema_version": "1.0", "intent": "scan_report", "timestamp": "2026-01-01T00:00:00+08:00", "parameters": {"query": "前方"}, "confidence": 0.9, "reasoning": "用户询问前方障碍物情况"}

对于无法理解或模糊的指令（如"开到那边去"），使用 clarify 意图并给出澄清问题。

只输出 JSON，不要输出任何其他内容。"""


class VLLMClient:
    """Async client for LLM inference (llama.cpp / vLLM OpenAI-compatible API).

    Currently configured for llama.cpp server.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API endpoint (default: llama.cpp local server).
    model : str
        Model name as registered in the server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen3-8B-Q4_K_M",
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._client = None  # Lazy init

    @property
    def _async_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self._base_url, api_key="not-needed")
        return self._client

    async def parse(self, text: str) -> dict | None:
        """Send text to the LLM and return parsed JSON dict.

        Returns None on parse failure or timeout.
        """
        import re

        try:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=512,
                extra_body={"enable_thinking": False},
                timeout=10.0,
            )
            content = response.choices[0].message.content
            if content is None:
                return None
            # Strip <think>...</think> tags if present (Qwen3 thinking mode)
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            # Strip markdown code fences if present
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            return json.loads(content)
        except Exception:
            return None
