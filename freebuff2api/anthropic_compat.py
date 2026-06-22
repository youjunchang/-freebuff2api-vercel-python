"""
Anthropic Messages API <-> OpenAI Chat Completions protocol conversion.
Allows freebuff2api to accept both Anthropic-format and OpenAI-format requests.
"""

from __future__ import annotations

import json
import time
from typing import Any


# ─── Anthropic Request → OpenAI Request ─────────────────────────────

def anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages API request to OpenAI Chat Completions format."""
    result: dict[str, Any] = {}

    # Copy model
    result["model"] = body.get("model")

    # Build messages list
    messages: list[dict[str, Any]] = []

    # Convert system prompt → system role message
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            texts = [
                b["text"]
                for b in system
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            if texts:
                messages.append({"role": "system", "content": "\n".join(texts)})

    # Convert message list
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            if role == "user":
                _convert_user_blocks(content, messages)
            elif role == "assistant":
                _convert_assistant_blocks(content, messages)
            else:
                messages.append({"role": role, "content": ""})

    result["messages"] = messages

    # Copy common parameters
    for key in ("max_tokens", "temperature", "top_p", "stream"):
        if key in body and body[key] is not None:
            result[key] = body[key]

    # stop_sequences → stop
    if body.get("stop_sequences"):
        result["stop"] = body["stop_sequences"]

    # tools
    if body.get("tools"):
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in body["tools"]
            if t.get("name")
        ]

    # tool_choice
    if "tool_choice" in body:
        result["tool_choice"] = _convert_tool_choice(body["tool_choice"])

    # thinking → reasoning_effort
    if body.get("thinking"):
        budget = body["thinking"].get("budget_tokens")
        if budget is not None:
            if budget <= 2048:
                result["reasoning_effort"] = "low"
            elif budget <= 8192:
                result["reasoning_effort"] = "medium"
            else:
                result["reasoning_effort"] = "high"
        elif body["thinking"].get("type") == "enabled":
            result["reasoning_effort"] = "high"

    return result


def _convert_user_blocks(
    blocks: list[dict[str, Any]],
    output: list[dict[str, Any]],
) -> None:
    """Convert Anthropic user content blocks (text/image/tool_result) to OpenAI messages."""
    content_parts: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")

        if block_type == "text":
            content_parts.append({"type": "text", "text": block["text"]})

        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
                content_parts.append({"type": "image_url", "image_url": {"url": url}})

        elif block_type == "tool_result":
            tool_content = block.get("content")
            if isinstance(tool_content, str):
                text = tool_content
            elif isinstance(tool_content, list):
                text = "\n".join(
                    b["text"]
                    for b in tool_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = ""
            output.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": text,
            })

    if content_parts:
        merged = (
            content_parts[0]["text"]
            if len(content_parts) == 1 and content_parts[0]["type"] == "text"
            else content_parts
        )
        output.append({"role": "user", "content": merged})


def _convert_assistant_blocks(
    blocks: list[dict[str, Any]],
    output: list[dict[str, Any]],
) -> None:
    """Convert Anthropic assistant content blocks (text/tool_use/thinking) to OpenAI message.

    Preserves thinking blocks as reasoning_content so the upstream API
    does not reject the request with 'reasoning_content must be passed back'.
    """
    text_content = ""
    thinking_content = ""
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_content += block.get("text", "")
        elif block_type == "thinking":
            thinking_content += block.get("thinking", "")
        elif block_type == "tool_use":
            raw_input = block.get("input", {})
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": (
                        raw_input
                        if isinstance(raw_input, str)
                        else json.dumps(raw_input, ensure_ascii=False)
                    ),
                },
            })

    msg: dict[str, Any] = {
        "role": "assistant",
        "content": text_content or None,
    }
    if thinking_content:
        msg["reasoning_content"] = thinking_content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    output.append(msg)


def _convert_tool_choice(choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI tool_choice."""
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        t = choice.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "none":
            return "none"
        if t == "tool" and choice.get("name"):
            return {"type": "function", "function": {"name": choice["name"]}}
    return "auto"


# ─── OpenAI Response → Anthropic Response ──────────────────────────

_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def openai_to_anthropic(
    openai_response: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format."""
    choice = (openai_response.get("choices") or [{}])[0]
    message = choice.get("message", {}) if choice else {}

    content: list[dict[str, Any]] = []

    # reasoning_content (DeepSeek, OpenRouter, etc.)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    # text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # tool calls
    for tc in message.get("tool_calls") or []:
        try:
            input_data = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError, KeyError):
            input_data = {"raw": tc["function"]["arguments"]}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc["function"]["name"],
            "input": input_data,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    # finish reason
    stop_reason = _FINISH_REASON_MAP.get(choice.get("finish_reason") if choice else None, "end_turn")

    # usage
    usage = openai_response.get("usage") or {}

    return {
        "id": openai_response.get("id") or f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": openai_response.get("model") or model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": (
                usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if isinstance(usage.get("prompt_tokens_details"), dict)
                else 0
            ),
        },
    }


# ─── OpenAI Streaming Chunk → Anthropic SSE Events ────────────────


class AnthropicStreamTransformer:
    """Transforms an OpenAI streaming chunk iterator into Anthropic SSE events.

    Handles text, thinking/reasoning, and tool_use blocks with proper
    block lifecycle (start → delta* → stop). Supports deferred finish_reason
    for DeepSeek-style APIs that send finish_reason before arguments arrive.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._message_id: str = ""
        self._has_started = False
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._stream_finished = False

        # Block tracking — open blocks are pushed/popped in order
        self._open_blocks: list[int] = []  # stack of open block indices
        self._next_block_index: int = -1

        # New block index allocator
        self._thinking_block_index: int | None = None
        self._text_block_index: int | None = None

        # Tool call tracking: OpenAI tool_call index → tool block state
        # ToolBlockState: {anthropic_index, id, name, args_buffer, started}
        self._tool_blocks: dict[int, dict[str, Any]] = {}
        self._has_tool_use = False

        # Deferred finish handling (DeepSeek sends finish_reason before args arrive)
        self._deferred_finish_reason: str | None = None
        self._deferred_output_tokens: int = 0

    # ── Block management ──────────────────────────────────────

    def _alloc_block_index(self) -> int:
        self._next_block_index += 1
        self._open_blocks.append(self._next_block_index)
        return self._next_block_index

    def _start_block(self, events: list[dict[str, Any]], block_type: str,
                     extra: dict[str, Any] | None = None) -> int:
        idx = self._alloc_block_index()
        block: dict[str, Any] = {"type": block_type}
        if extra:
            block.update(extra)
        events.append({
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": idx,
                "content_block": block,
            },
        })
        return idx

    def _close_all_blocks(self, events: list[dict[str, Any]]) -> None:
        for idx in reversed(self._open_blocks):
            events.append({
                "event": "content_block_stop",
                "data": {"type": "content_block_stop", "index": idx},
            })
        self._open_blocks.clear()

    def _emit_text_delta(self, events: list[dict[str, Any]], text: str) -> None:
        idx = self._text_block_index if self._text_block_index is not None else (
            self._open_blocks[-1] if self._open_blocks else self._alloc_block_index())
        events.append({
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": text},
            },
        })

    def _emit_thinking_delta(self, events: list[dict[str, Any]],
                             thinking: str) -> None:
        events.append({
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": self._thinking_block_index or 0,
                "delta": {"type": "thinking_delta", "thinking": thinking},
            },
        })

    def _emit_tool_args_delta(self, events: list[dict[str, Any]],
                              block_idx: int, partial_json: str) -> None:
        events.append({
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "input_json_delta", "partial_json": partial_json},
            },
        })

    def _emit_finish_events(self, events: list[dict[str, Any]],
                            stop_reason: str) -> None:
        self._close_all_blocks(events)
        fr_str = f'"{stop_reason}"' if stop_reason else "null"
        events.append({
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason or None,
                          "stop_sequence": None},
                "usage": {"output_tokens": self._output_tokens},
            },
        })
        events.append({
            "event": "message_stop",
            "data": {"type": "message_stop"},
        })

    # ── Public API ───────────────────────────────────────────

    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Process one OpenAI streaming chunk, return list of Anthropic SSE event dicts.
        Each event dict:: {"event": str, "data": dict}
        """
        events: list[dict[str, Any]] = []
        choices = chunk.get("choices") or []
        delta = choices[0].get("delta", {}) if choices else {}
        finish_reason = choices[0].get("finish_reason") if choices else None

        # Track IDs and model
        self._message_id = chunk.get("id") or self._message_id

        # Usage may appear in any chunk (esp. final one)
        if chunk.get("usage"):
            usage = chunk["usage"]
            self._input_tokens = usage.get("prompt_tokens", 0)
            self._output_tokens = usage.get("completion_tokens", 0)

        content = delta.get("content") if isinstance(delta, dict) else None
        reasoning = delta.get("reasoning_content") if isinstance(delta, dict) else None
        raw_tool_calls = delta.get("tool_calls") if isinstance(delta, dict) else None

        # --- message_start (once) ---
        if not self._has_started and (
                delta.get("role") == "assistant"
                or content is not None
                or reasoning is not None
                or raw_tool_calls is not None
        ):
            self._has_started = True
            events.append(self._message_start_event())

        # --- reasoning / thinking content ---
        if reasoning:
            if self._thinking_block_index is None:
                self._thinking_block_index = self._start_block(
                    events, "thinking", {"thinking": ""})
            self._emit_thinking_delta(events, reasoning)

        # --- text content ---
        # Only open a text block when there's actual content and we're not
        # inside a tool_use block that should own the text.
        if content is not None and content:
            # Close thinking block if open
            if self._thinking_block_index is not None and self._thinking_block_index in self._open_blocks:
                self._close_all_blocks(events)
                self._thinking_block_index = -1  # mark as closed

            if self._text_block_index is None:
                self._text_block_index = self._start_block(
                    events, "text", {"text": ""})
            self._emit_text_delta(events, content)

        # --- tool calls ---
        if raw_tool_calls is not None:
            for tc_raw in raw_tool_calls:
                tc_index = tc_raw.get("index", 0)
                fn = tc_raw.get("function", {})
                tc_id = tc_raw.get("id", "")
                tc_name = fn.get("name", "") if isinstance(fn, dict) else ""
                tc_args = fn.get("arguments", "") if isinstance(fn, dict) else ""

                # First tool call ever — close all open text/thinking blocks
                if not self._has_tool_use:
                    self._has_tool_use = True
                    self._close_all_blocks(events)

                if tc_index not in self._tool_blocks:
                    block: dict[str, Any] = {
                        "anthropic_index": -1,
                        "id": tc_id,
                        "name": tc_name or "",
                        "args_buffer": tc_args or "",
                        "started": False,
                    }
                    self._tool_blocks[tc_index] = block
                else:
                    block = self._tool_blocks[tc_index]
                    if tc_id:
                        block["id"] = tc_id
                    if tc_name:
                        block["name"] += tc_name
                    if tc_args:
                        block["args_buffer"] += tc_args

                b = self._tool_blocks[tc_index]
                # Start tool_use content block once we have id + name
                if not b["started"] and b["id"] and b["name"]:
                    b["started"] = True
                    b["anthropic_index"] = self._alloc_block_index()
                    events.append({
                        "event": "content_block_start",
                        "data": {
                            "type": "content_block_start",
                            "index": b["anthropic_index"],
                            "content_block": {
                                "type": "tool_use",
                                "id": b["id"],
                                "name": b["name"],
                                "input": {},
                            },
                        },
                    })
                    # Flush buffered arguments from the same chunk
                    if b["args_buffer"]:
                        self._emit_tool_args_delta(
                            events, b["anthropic_index"], b["args_buffer"])
                elif b["started"] and tc_args:
                    self._emit_tool_args_delta(
                        events, b["anthropic_index"], tc_args)

        # --- finish reason ---
        if finish_reason:
            # DeepSeek sends finish_reason="tool_calls" BEFORE arguments arrive.
            # Defer the finish signal when tool_use blocks are still accumulating args.
            if (finish_reason == "tool_calls"
                    and self._has_tool_use
                    and len(self._open_blocks) > 0):
                self._deferred_finish_reason = finish_reason
                self._deferred_output_tokens = self._output_tokens
            else:
                self._stream_finished = True
                self._emit_finish_events(
                    events, _FINISH_REASON_MAP.get(finish_reason, "end_turn"))

        return events

    def flush(self) -> list[dict[str, Any]]:
        """Emit closing events if the stream ended without finish_reason."""
        events: list[dict[str, Any]] = []

        if self._stream_finished:
            return events

        # Handle deferred finish (DeepSeek tool_calls)
        if self._deferred_finish_reason:
            self._output_tokens = self._deferred_output_tokens
            self._emit_finish_events(
                events, _FINISH_REASON_MAP.get(self._deferred_finish_reason, "end_turn"))
            self._stream_finished = True
            self._deferred_finish_reason = None
            return events

        if not self._has_started:
            events.append(self._message_start_event())

        if self._has_started:
            self._close_all_blocks(events)
            events.append({
                "event": "message_delta",
                "data": {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": self._output_tokens},
                },
            })
            events.append({
                "event": "message_stop",
                "data": {"type": "message_stop"},
            })

        return events

    def _message_start_event(self) -> dict[str, Any]:
        return {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": self._message_id or f"msg_{int(time.time() * 1000)}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self._input_tokens,
                        "output_tokens": 0,
                    },
                },
            },
        }
