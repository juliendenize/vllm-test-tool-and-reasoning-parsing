"""
Standalone test script for vLLM tool calling WITHOUT reasoning_effort.

Tests tool_choice modes (auto, required, named, none) with and without
streaming against a running vLLM-compatible server.  The reasoning_effort
parameter is **never** sent, so the model decides on its own when to
reason.  For prompts that produce textual content the test expects reasoning
to be present; for prompts that only produce tool calls, reasoning presence
is not enforced.

Prints a colored summary table and writes detailed results (tool calls,
reasoning trace, content) to a JSON file.

Usage:
    python test_vllm_tools_pre_v15.py --base-url http://localhost:8000/v1
    python test_vllm_tools_pre_v15.py --base-url http://localhost:8000/v1 --model my-model
    python test_vllm_tools_pre_v15.py --base-url http://localhost:8000/v1 -o results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_SUPPORTS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _SUPPORTS_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------
def load_system_prompt(repo_id: str, filename: str) -> dict[str, Any]:
    """Download and parse a system prompt file from Hugging Face Hub.

    The file is expected to contain ``[THINK]...[/THINK]`` markers that
    delimit a *thinking* section.  The returned message dict uses the
    multi-part ``content`` format so that vLLM can inject the thinking
    block as pre-filled reasoning.
    """
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()

    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": (
                        "The two-letter abbreviation for the state that the city is in, "
                        "e.g. 'CA' for California"
                    ),
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state"],
        },
    },
}

SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet and get a summary of the top 10 webpages. "
            "Should only be used if you don't know the answer to a user query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Keywords to search for",
                }
            },
            "required": ["search_term"],
        },
    },
}

ALL_TOOLS = [WEATHER_TOOL, SEARCH_TOOL]

# ---------------------------------------------------------------------------
# Messages for different scenarios
# ---------------------------------------------------------------------------
# Should trigger a tool call (weather)
MESSAGES_WANT_TOOL: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": "What is the weather in Dallas, Texas in Fahrenheit?",
    }
]

# Should NOT trigger a tool call
MESSAGES_NO_TOOL: list[dict[str, Any]] = [
    {"role": "user", "content": "Tell me a short joke about programming."}
]

# Tool result follow-up (model should respond with content, not more tools)
MESSAGES_TOOL_RESULT: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": "What is the weather in Dallas, Texas in Fahrenheit?",
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "chatcmpl0",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": '{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "chatcmpl0",
        "content": (
            "The weather in Dallas is 98 degrees Fahrenheit, with partly "
            "cloudy skies and a low chance of rain."
        ),
    },
]


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    """A single test scenario."""

    name: str
    description: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any]
    stream: bool
    reasoning_effort: str | None  # Always None in this script
    # Expectations
    expect_tool_calls: bool | None  # True = must have, False = must not, None = either
    expect_content: bool | None  # True = must have, False = must not, None = either
    expect_finish_reason: str | None  # expected finish_reason, None = don't check
    expect_reasoning: bool | None  # True = must have, False = must not, None = either


def build_test_cases() -> list[TestCase]:
    """Build the full test matrix.

    reasoning_effort is never sent.  The model decides on its own when to
    reason.  We expect reasoning when the model produces textual content,
    and leave it unchecked (None) when it produces only tool calls.
    """
    cases: list[TestCase] = []

    for stream in (False, True):
        sfx = " (stream)" if stream else " (non-stream)"

        # --- tool_choice = "auto" ---
        # Prompt that SHOULD trigger tools
        cases.append(
            TestCase(
                name=f"auto_with_tool_prompt{sfx}",
                description=f"tool_choice=auto, prompt wants weather{sfx}",
                messages=MESSAGES_WANT_TOOL,
                tools=ALL_TOOLS,
                tool_choice="auto",
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=True,
                expect_content=None,
                expect_finish_reason="tool_calls",
                expect_reasoning=None,  # tool-only: model may or may not reason
            )
        )
        # Prompt that should NOT trigger tools
        cases.append(
            TestCase(
                name=f"auto_no_tool_prompt{sfx}",
                description=f"tool_choice=auto, prompt wants joke{sfx}",
                messages=MESSAGES_NO_TOOL,
                tools=ALL_TOOLS,
                tool_choice="auto",
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=False,
                expect_content=True,
                expect_finish_reason="stop",
                expect_reasoning=True,  # content prompt: expect reasoning
            )
        )
        # Tool result follow-up: ideally the model answers with content,
        # but without reasoning_effort the model may aggressively re-call
        # tools instead of synthesizing the tool result into an answer.
        # We accept either behaviour.
        cases.append(
            TestCase(
                name=f"auto_tool_result{sfx}",
                description=f"tool_choice=auto, tool result follow-up{sfx}",
                messages=MESSAGES_TOOL_RESULT,
                tools=ALL_TOOLS,
                tool_choice="auto",
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=None,
                expect_content=None,
                expect_finish_reason=None,
                expect_reasoning=None,  # model may or may not reason
            )
        )

        # --- tool_choice = "required" ---
        cases.append(
            TestCase(
                name=f"required_tool_prompt{sfx}",
                description=f"tool_choice=required, prompt wants weather{sfx}",
                messages=MESSAGES_WANT_TOOL,
                tools=ALL_TOOLS,
                tool_choice="required",
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=True,
                expect_content=None,
                expect_finish_reason="tool_calls",
                expect_reasoning=None,  # tool-only: model may or may not reason
            )
        )

        # --- tool_choice = named (specific function) ---
        # Per the OpenAI API spec, finish_reason is "stop" for
        # named tool calls (not "tool_calls").
        cases.append(
            TestCase(
                name=f"named_weather{sfx}",
                description=f"tool_choice=get_current_weather (named){sfx}",
                messages=MESSAGES_WANT_TOOL,
                tools=ALL_TOOLS,
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_current_weather"},
                },
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=True,
                expect_content=None,
                expect_finish_reason="stop",
                expect_reasoning=None,  # tool-only: model may or may not reason
            )
        )
        # Named tool but prompt doesn't naturally want it
        cases.append(
            TestCase(
                name=f"named_search_wrong_prompt{sfx}",
                description=f"tool_choice=web_search (named), weather prompt{sfx}",
                messages=MESSAGES_WANT_TOOL,
                tools=ALL_TOOLS,
                tool_choice={
                    "type": "function",
                    "function": {"name": "web_search"},
                },
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=True,
                expect_content=None,
                expect_finish_reason="stop",
                expect_reasoning=None,  # tool-only: model may or may not reason
            )
        )

        # --- tool_choice = "none" ---
        # For Mistral models, tool_choice=none now injects a Lark grammar
        # that constrains output to ``think? content`` (no tool calls).
        # Whether reasoning is present depends on the tokenizer version
        # and reasoning_effort setting, so we don't enforce it.
        cases.append(
            TestCase(
                name=f"none_tool_prompt{sfx}",
                description=f"tool_choice=none, prompt wants weather{sfx}",
                messages=MESSAGES_WANT_TOOL,
                tools=ALL_TOOLS,
                tool_choice="none",
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=False,
                expect_content=True,
                expect_finish_reason=None,
                expect_reasoning=None,  # depends on tokenizer version
            )
        )
        cases.append(
            TestCase(
                name=f"none_no_tool_prompt{sfx}",
                description=f"tool_choice=none, prompt wants joke{sfx}",
                messages=MESSAGES_NO_TOOL,
                tools=ALL_TOOLS,
                tool_choice="none",
                stream=stream,
                reasoning_effort=None,
                expect_tool_calls=False,
                expect_content=True,
                expect_finish_reason=None,
                expect_reasoning=None,  # depends on tokenizer version
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    description: str
    tool_choice: str | dict
    stream: bool
    reasoning_effort: str | None  # Always None in this script
    passed: bool = False
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None
    # Raw response data
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str | None = None
    finish_reason: str | None = None
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Streaming reconstruction helpers
# ---------------------------------------------------------------------------
def reconstruct_streaming(
    chunks: list,
) -> tuple[str | None, list[dict[str, Any]], str | None, str | None]:
    """Reconstruct content, tool_calls, reasoning_content, and finish_reason
    from a list of streaming ChatCompletionChunk objects."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_map: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
    finish_reason: str | None = None

    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        delta = choice.delta

        # Content
        if delta.content:
            content_parts.append(delta.content)

        # Reasoning content — vLLM uses the field name ``reasoning``
        rc = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if rc:
            reasoning_parts.append(rc)

        # Tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_map:
                    tool_map[idx] = {"id": None, "name": None, "arguments": ""}
                if tc.id:
                    tool_map[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_map[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_map[idx]["arguments"] += tc.function.arguments

    content = "".join(content_parts) if content_parts else None
    reasoning = "".join(reasoning_parts) if reasoning_parts else None

    tool_calls_list: list[dict[str, Any]] = []
    for idx in sorted(tool_map.keys()):
        tc = tool_map[idx]
        entry: dict[str, Any] = {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"],
            },
        }
        # Try to parse arguments as JSON
        try:
            entry["function"]["arguments_parsed"] = json.loads(tc["arguments"])
        except (json.JSONDecodeError, TypeError):
            entry["function"]["arguments_parsed"] = None
        tool_calls_list.append(entry)

    return content, tool_calls_list, reasoning, finish_reason


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------
async def run_single_test(
    client: AsyncOpenAI,
    model: str,
    tc: TestCase,
) -> TestResult:
    """Run a single test case and return the result."""
    result = TestResult(
        name=tc.name,
        description=tc.description,
        tool_choice=tc.tool_choice
        if isinstance(tc.tool_choice, str)
        else tc.tool_choice,
        stream=tc.stream,
        reasoning_effort=tc.reasoning_effort,
    )

    t0 = time.monotonic()
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": tc.messages,
            "tools": tc.tools,
            "tool_choice": tc.tool_choice,
            "temperature": 0,
            "max_completion_tokens": 16384,
            "stream": tc.stream,
        }

        # reasoning_effort is never sent in this script (always None).

        if tc.stream:
            stream = await client.chat.completions.create(**kwargs)
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            content, tool_calls, reasoning, finish_reason = reconstruct_streaming(
                chunks
            )
            result.content = content
            result.tool_calls = tool_calls
            result.reasoning_content = reasoning
            result.finish_reason = finish_reason
        else:
            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            result.content = choice.message.content
            result.finish_reason = choice.finish_reason

            # Reasoning content — vLLM uses the field name ``reasoning``
            rc = getattr(choice.message, "reasoning", None) or getattr(
                choice.message, "reasoning_content", None
            )
            if rc:
                result.reasoning_content = rc

            # Tool calls
            if choice.message.tool_calls:
                for tc_obj in choice.message.tool_calls:
                    entry: dict[str, Any] = {
                        "id": tc_obj.id,
                        "type": tc_obj.type,
                        "function": {
                            "name": tc_obj.function.name,
                            "arguments": tc_obj.function.arguments,
                        },
                    }
                    try:
                        entry["function"]["arguments_parsed"] = json.loads(
                            tc_obj.function.arguments
                        )
                    except (json.JSONDecodeError, TypeError):
                        entry["function"]["arguments_parsed"] = None
                    result.tool_calls.append(entry)

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        result.duration_s = time.monotonic() - t0
        return result

    result.duration_s = time.monotonic() - t0

    # ----- Validation checks -----
    all_ok = True

    # Check: no HTTP / server error
    result.checks["no_error"] = {"expected": "no error", "actual": "ok", "passed": True}
    if result.error:
        result.checks["no_error"] = {
            "expected": "no error",
            "actual": result.error[:120],
            "passed": False,
        }
        all_ok = False

    has_tool_calls = len(result.tool_calls) > 0
    has_content = result.content is not None and len(result.content.strip()) > 0

    # Check: tool_calls presence
    if tc.expect_tool_calls is True:
        ok = has_tool_calls
        result.checks["tool_calls_present"] = {
            "expected": "tool calls present",
            "actual": f"{len(result.tool_calls)} tool call(s)"
            if ok
            else "no tool calls",
            "passed": ok,
        }
        if not ok:
            all_ok = False

        # If tool calls expected, validate structure
        if has_tool_calls:
            for i, tcc in enumerate(result.tool_calls):
                fn = tcc.get("function", {})
                name_ok = fn.get("name") is not None and isinstance(fn.get("name"), str)
                args_ok = fn.get("arguments") is not None
                parsed_ok = fn.get("arguments_parsed") is not None
                check_ok = name_ok and args_ok and parsed_ok
                result.checks[f"tool_call_{i}_valid"] = {
                    "expected": "valid function name + parseable JSON args",
                    "actual": (f"name={fn.get('name')!r}, args_parseable={parsed_ok}"),
                    "passed": check_ok,
                }
                if not check_ok:
                    all_ok = False

    elif tc.expect_tool_calls is False:
        ok = not has_tool_calls
        result.checks["no_tool_calls"] = {
            "expected": "no tool calls",
            "actual": "no tool calls"
            if ok
            else f"{len(result.tool_calls)} tool call(s)",
            "passed": ok,
        }
        if not ok:
            all_ok = False

    # Check: content presence
    if tc.expect_content is True:
        ok = has_content
        result.checks["content_present"] = {
            "expected": "content present",
            "actual": (
                f"content ({len(result.content)} chars)" if ok else "no content"
            ),
            "passed": ok,
        }
        if not ok:
            all_ok = False
    elif tc.expect_content is False:
        ok = not has_content
        result.checks["no_content"] = {
            "expected": "no content",
            "actual": "no content" if ok else f"content ({len(result.content)} chars)",
            "passed": ok,
        }
        if not ok:
            all_ok = False

    # Check: finish_reason
    if tc.expect_finish_reason is not None:
        ok = result.finish_reason == tc.expect_finish_reason
        result.checks["finish_reason"] = {
            "expected": tc.expect_finish_reason,
            "actual": result.finish_reason,
            "passed": ok,
        }
        if not ok:
            all_ok = False

    # Check: named tool_choice forces the correct function name
    if isinstance(tc.tool_choice, dict) and has_tool_calls:
        expected_name = tc.tool_choice["function"]["name"]
        actual_name = result.tool_calls[0]["function"].get("name")
        ok = actual_name == expected_name
        result.checks["named_tool_match"] = {
            "expected": f"function name = {expected_name!r}",
            "actual": f"function name = {actual_name!r}",
            "passed": ok,
        }
        if not ok:
            all_ok = False

    # Check: reasoning content presence
    has_reasoning = (
        result.reasoning_content is not None
        and len(result.reasoning_content.strip()) > 0
    )
    if tc.expect_reasoning is True:
        ok = has_reasoning
        result.checks["reasoning_present"] = {
            "expected": "reasoning content present",
            "actual": (
                f"reasoning ({len(result.reasoning_content)} chars)"
                if ok
                else "no reasoning content"
            ),
            "passed": ok,
        }
        if not ok:
            all_ok = False
    elif tc.expect_reasoning is False:
        ok = not has_reasoning
        result.checks["no_reasoning"] = {
            "expected": "no reasoning content",
            "actual": (
                "no reasoning content"
                if ok
                else f"reasoning ({len(result.reasoning_content)} chars)"
            ),
            "passed": ok,
        }
        if not ok:
            all_ok = False

    result.passed = all_ok
    return result


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_result(result: TestResult, index: int, total: int) -> None:
    status = green("PASS") if result.passed else red("FAIL")
    stream_tag = cyan("stream") if result.stream else dim("no-stream")

    tc_str = (
        result.tool_choice
        if isinstance(result.tool_choice, str)
        else result.tool_choice["function"]["name"]
    )

    print(
        f"\n{'=' * 72}\n"
        f"[{index}/{total}] {status}  {bold(result.name)}\n"
        f"  {result.description}\n"
        f"  tool_choice={tc_str}  {stream_tag}  "
        f"duration={result.duration_s:.2f}s"
    )

    if result.error:
        print(f"  {red('ERROR:')} {result.error[:200]}")

    for check_name, check_info in result.checks.items():
        icon = green("OK") if check_info["passed"] else red("FAIL")
        print(
            f"    [{icon}] {check_name}: "
            f"expected={check_info['expected']}, "
            f"actual={check_info['actual']}"
        )

    # Show a snippet of content / tool calls
    if result.content:
        snippet = result.content[:150].replace("\n", " ")
        if len(result.content) > 150:
            snippet += "..."
        print(f"  {dim('content:')} {snippet}")
    if result.reasoning_content:
        snippet = result.reasoning_content[:150].replace("\n", " ")
        if len(result.reasoning_content) > 150:
            snippet += "..."
        print(f"  {dim('reasoning:')} {snippet}")
    if result.tool_calls:
        for i, tc in enumerate(result.tool_calls):
            fn = tc.get("function", {})
            print(
                f"  {dim(f'tool_call[{i}]:')} "
                f"{fn.get('name')}({fn.get('arguments', '')[:100]})"
            )


def print_summary_table(results: list[TestResult]) -> None:
    """Print a nice ASCII summary table."""
    print(f"\n\n{'=' * 80}")
    print(bold("                              SUMMARY TABLE"))
    print(f"{'=' * 80}")

    # Column widths
    col_name = 38
    col_tc = 22
    col_mode = 11
    col_status = 8
    col_dur = 8

    header = (
        f"{'Test Name':<{col_name}} "
        f"{'Tool Choice':<{col_tc}} "
        f"{'Mode':<{col_mode}} "
        f"{'Status':<{col_status}} "
        f"{'Time':<{col_dur}}"
    )
    print(bold(header))
    print("-" * 80)

    for r in results:
        tc_str = (
            r.tool_choice
            if isinstance(r.tool_choice, str)
            else r.tool_choice["function"]["name"]
        )
        mode = "stream" if r.stream else "non-stream"
        status = green("PASS") if r.passed else red("FAIL")
        dur = f"{r.duration_s:.2f}s"

        # Truncate name if needed
        name = r.name
        if len(name) > col_name:
            name = name[: col_name - 3] + "..."

        print(
            f"{name:<{col_name}} "
            f"{tc_str:<{col_tc}} "
            f"{mode:<{col_mode}} "
            f"{status:<{col_status + 9}} "  # +9 for ANSI escape codes
            f"{dur:<{col_dur}}"
        )

    print("-" * 80)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    total_time = sum(r.duration_s for r in results)

    summary_color = green if failed == 0 else red
    print(
        summary_color(
            f"\n  {passed}/{total} passed, {failed} failed  "
            f"(total time: {total_time:.2f}s)"
        )
    )

    if failed > 0:
        print(f"\n  {red('Failed tests:')}")
        for r in results:
            if not r.passed:
                failed_checks = [k for k, v in r.checks.items() if not v["passed"]]
                print(f"    - {r.name}: {', '.join(failed_checks)}")
                if r.error:
                    print(f"      error: {r.error[:150]}")

    print()


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------
def export_results(
    results: list[TestResult], path: str, model: str, base_url: str
) -> None:
    """Write detailed results to a JSON file."""
    data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "base_url": base_url,
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "reasoning_effort": "not set (model decides)",
        },
        "results": [],
    }

    for r in results:
        entry = {
            "name": r.name,
            "description": r.description,
            "tool_choice": r.tool_choice,
            "stream": r.stream,
            "reasoning_effort": r.reasoning_effort,
            "passed": r.passed,
            "duration_s": r.duration_s,
            "finish_reason": r.finish_reason,
            "checks": r.checks,
            "content": r.content,
            "reasoning_content": r.reasoning_content,
            "tool_calls": r.tool_calls,
            "error": r.error,
        }
        data["results"].append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nDetailed results written to {bold(path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test vLLM tool calling (auto / required / named / none, "
            "streaming & non-streaming) WITHOUT reasoning_effort. "
            "The model decides on its own when to reason."
        ),
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the vLLM-compatible server (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use. If omitted, the first model from /v1/models is used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="vllm_tool_test_pre_v15_results.json",
        help="Path to the output JSON file (default: vllm_tool_test_pre_v15_results.json)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the server (default: EMPTY)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run tests whose name contains this substring",
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    # Resolve model name
    model = args.model
    if model is None:
        print("Fetching model list from server...")
        models = await client.models.list()
        if not models.data:
            print(red("ERROR: No models available on the server."))
            return 1
        model = models.data[0].id
    print(f"Using model: {bold(model)}")
    print(f"Server:      {args.base_url}")
    print(dim("reasoning_effort: not set (model decides when to reason)"))

    # Load system prompt from the model's HF repo
    print("Loading system prompt from Hugging Face Hub...")
    system_prompt = load_system_prompt(model, "SYSTEM_PROMPT.txt")
    print(f"System prompt loaded ({dim('with [THINK] block')})")
    print()

    # Build test cases
    test_cases = build_test_cases()
    if args.filter:
        test_cases = [tc for tc in test_cases if args.filter in tc.name]
        print(f"Filter '{args.filter}' matched {len(test_cases)} test(s)")

    if not test_cases:
        print(yellow("No test cases to run."))
        return 0

    # Prepend the system prompt to every test case's messages
    for tc in test_cases:
        tc.messages = [system_prompt] + tc.messages

    print(f"Running {bold(str(len(test_cases)))} test cases...\n")

    # Run tests sequentially (to avoid overloading the server and to get
    # clean deterministic output)
    results: list[TestResult] = []
    for i, tc in enumerate(test_cases, 1):
        result = await run_single_test(client, model, tc)
        results.append(result)
        print_result(result, i, len(test_cases))

    # Summary
    print_summary_table(results)

    # Export
    export_results(results, args.output, model, args.base_url)

    # Exit code
    failed = sum(1 for r in results if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
