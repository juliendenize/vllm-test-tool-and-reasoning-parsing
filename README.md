# vLLM Test Tool and Reasoning Parsing

Small test scripts for validating **reasoning** and **tool call parsing** behavior of Mistral models served through vLLM's OpenAI-compatible API.

> **Disclaimer:** This repository was vibe-coded and has not been thoroughly tested nor cleaned.

## Context

Mistral models use a Lark grammar to structure their output into `think? (content | fcalls)` patterns. The grammar behavior differs depending on the **tokenizer version**:

- **Pre-v15 tokenizer** (`test_vllm_tools_pre_v15.py`): Does not support the `reasoning_effort` parameter. The model decides autonomously whether to produce `[THINK]...[/THINK]` blocks. A system prompt with pre-filled thinking context is downloaded from the model's Hugging Face repository.
- **Post-v15 tokenizer** (`test_vllm_tools_post_v15.py`): Supports the `reasoning_effort` parameter (a vLLM extension passed via `extra_body`), which explicitly controls whether the model produces reasoning traces (`"high"`, `"none"`, or omitted).

## Scripts

### `test_vllm_tools_post_v15.py` — With `reasoning_effort` support (post-v15)

Tests tool calling across three dimensions:

| Dimension | Values |
|---|---|
| `tool_choice` | `"auto"`, `"required"`, named (specific function), `"none"` |
| `stream` | `True`, `False` |
| `reasoning_effort` | `None` (omitted), `"none"` (disable thinking), `"high"` (enable thinking) |

This produces **48 test cases** (8 scenarios x 2 stream modes x 3 reasoning_effort values), or **16** when run with `--no-reasoning-effort`.

```bash
python test_vllm_tools_post_v15.py --base-url http://localhost:8000/v1
python test_vllm_tools_post_v15.py --base-url http://localhost:8000/v1 --model my-model
python test_vllm_tools_post_v15.py --base-url http://localhost:8000/v1 --no-reasoning-effort
```

### `test_vllm_tools_pre_v15.py` — Without `reasoning_effort` (pre-v15)

Same test scenarios but `reasoning_effort` is never sent. The model decides on its own whether to reason. A system prompt is fetched from the model's Hugging Face repo via `hf_hub_download`, and `[THINK]...[/THINK]` blocks in it are parsed into structured `{"type": "thinking"}` message parts.

This produces **16 test cases** (8 scenarios x 2 stream modes).

```bash
python test_vllm_tools_pre_v15.py --base-url http://localhost:8000/v1
python test_vllm_tools_pre_v15.py --base-url http://localhost:8000/v1 --model my-model
```

## Test Scenarios

Each script runs the following 8 scenarios per `(stream, reasoning_effort)` combination:

| # | Name | `tool_choice` | Prompt | Expected Behavior |
|---|---|---|---|---|
| 1 | `auto_with_tool_prompt` | `"auto"` | Weather query | Tool call |
| 2 | `auto_no_tool_prompt` | `"auto"` | Joke request | Content only, no tools |
| 3 | `auto_tool_result` | `"auto"` | Multi-turn with tool result | Content answer |
| 4 | `required_tool_prompt` | `"required"` | Weather query | Must call a tool |
| 5 | `named_weather` | `{"function": {"name": "get_current_weather"}}` | Weather query | Must call `get_current_weather` |
| 6 | `named_search_wrong_prompt` | `{"function": {"name": "web_search"}}` | Weather query | Must call `web_search` despite mismatch |
| 7 | `none_tool_prompt` | `"none"` | Weather query | Content only, no tools |
| 8 | `none_no_tool_prompt` | `"none"` | Joke request | Content only, no tools |

Two tools are defined for all tests: `get_current_weather` (city, state, unit) and `web_search` (search_term).

### Validation checks

Each test verifies:

- **`no_error`** — No HTTP/server error occurred.
- **`tool_calls_present`** / **`no_tool_calls`** — Tool calls are present or absent as expected.
- **`tool_call_N_valid`** — Function name is valid and arguments are parseable JSON.
- **`content_present`** / **`no_content`** — Content presence/absence matches expectations.
- **`finish_reason`** — Matches expected value (`"stop"` or `"tool_calls"`).
- **`named_tool_match`** — Named `tool_choice` forces the correct function name.
- **`reasoning_present`** / **`no_reasoning`** — Reasoning content presence/absence (when applicable).

## JSON Output

Results are exported as JSON files. The default file names are:

| Script | Default Output File |
|---|---|
| `test_vllm_tools_post_v15.py` | `vllm_tool_test_post_v15_results.json` |
| `test_vllm_tools_pre_v15.py` | `vllm_tool_test_pre_v15_results.json` |

Both can be overridden with `-o` / `--output`.

### JSON structure

```json
{
  "metadata": {
    "timestamp": "2026-03-14T21:41:39.987191+00:00",
    "model": "mistralai/Mistral-Small-4-119B-2603",
    "base_url": "http://localhost:8000/v1",
    "total_tests": 48,
    "passed": 48,
    "failed": 0
  },
  "results": [
    {
      "name": "auto_with_tool_prompt (non-stream, re=None)",
      "description": "tool_choice=auto, prompt wants weather (non-stream, re=None)",
      "tool_choice": "auto",
      "stream": false,
      "reasoning_effort": null,
      "passed": true,
      "duration_s": 1.38,
      "finish_reason": "tool_calls",
      "checks": {
        "no_error": { "expected": true, "actual": true, "passed": true },
        "tool_calls_present": { "expected": true, "actual": true, "passed": true }
      },
      "content": null,
      "reasoning_content": "Let me look up the weather...",
      "tool_calls": [
        {
          "id": "j95y80ix2",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\"city\": \"Dallas\", \"state\": \"TX\", \"unit\": \"fahrenheit\"}",
            "arguments_parsed": { "city": "Dallas", "state": "TX", "unit": "fahrenheit" }
          }
        }
      ],
      "error": null
    }
  ]
}
```

Test names in the JSON follow the pattern `{scenario} ({stream_mode}[, re={value}])`, for example:
- `"auto_with_tool_prompt (non-stream, re=None)"`
- `"named_weather (stream, re=high)"`
- `"none_no_tool_prompt (stream)"` (no `re=` suffix in the pre-v15 variant)

## CLI Options

Both scripts share common options:

| Flag | Description |
|---|---|
| `--base-url` | vLLM server URL (required), e.g. `http://localhost:8000/v1` |
| `--model` | Model name (defaults to server's loaded model) |
| `-o` / `--output` | Output JSON file path |
| `--api-key` | API key (default: `"token-abc123"`) |
| `--filter` | Filter test cases by name substring |

`test_vllm_tools_post_v15.py` additionally supports:

| Flag | Description |
|---|---|
| `--no-reasoning-effort` | Skip `reasoning_effort` dimension (only run with `None`) |
