"""
Responses from an OpenAI-compatible chat completions API (OpenRouter, vLLM, ...) in place of a local
model, for the trace scripts: the prompts are still templated and tokenized locally with the target's
tokenizer, the API supplies the assistant's reasoning and content as text, and the response token stream
is reconstructed by rendering the finished turn through the same chat template. Reasoning comes back under
whichever field the server uses (`reasoning`, `reasoning_content`, `reasoning_details`, or inline think tags).
"""
import os
import time
import json
import threading
import requests

# What's kept of a response for sending it back: providers want a tool call turn returned as they gave it
MESSAGE_FIELDS = ("role", "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details")


def resolve_api_key(explicit: str | None, api_base: str) -> str | None:
    if explicit:
        return explicit
    for env in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(env):
            return os.environ[env]
    path = os.path.expanduser("~/.config/openrouter_key")
    if "openrouter" in api_base and os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def split_inline_reasoning(content: str) -> tuple[str | None, str]:
    """Reasoning the server left inline as <think>...</think> at the start of the content"""
    if content and content.lstrip().startswith("<think>") and "</think>" in content:
        head, tail = content.lstrip()[len("<think>"):].split("</think>", 1)
        return head.strip("\n"), tail.lstrip("\n")
    return None, content


def extract_reasoning(message: dict) -> str | None:
    r = message.get("reasoning") or message.get("reasoning_content")
    if r:
        return r
    details = message.get("reasoning_details")
    if details:
        parts = [d.get("text") or d.get("summary") or "" for d in details if isinstance(d, dict)]
        parts = [p for p in parts if p]
        if parts:
            return "\n".join(parts)
    return None


class ApiChat:
    """One chat completion per call, with retries on rate limits and server errors"""

    def __init__(self, api_base: str, api_key: str | None, model: str, temperature: float,
                 max_tokens: int, extra: dict | None = None, retries: int = 6, timeout: float = 600.0):
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = extra or {}
        self.retries = retries
        self.timeout = timeout
        self.lock = threading.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "retries": 0}

    def request_reasoning(self, template_vars: dict):
        """OpenRouter only returns reasoning when asked: follow the template's thinking setting unless the
        request body already says"""
        if "reasoning" in self.extra or "include_reasoning" in self.extra or "openrouter" not in self.url \
                or not template_vars.get("enable_thinking", True):
            return
        effort = template_vars.get("reasoning_effort")
        self.extra["reasoning"] = {"effort": effort} if effort in ("low", "medium", "high") else {"enabled": True}

    def print_usage(self):
        u = self.usage
        print(f" -- API usage: {u['requests']} requests ({u['retries']} retries), {u['prompt_tokens']:,} prompt + "
              f"{u['completion_tokens']:,} completion tokens billed")

    def complete(self, messages: list, seed: int | None = None, tools: list | None = None) -> dict:
        """Returns {"content", "reasoning", "tool_calls", "finish_reason", "message"}, the last being the response
        as it can be sent back; raises after the last retry"""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
        body.update(self.extra)
        delay = 2.0
        for attempt in range(self.retries):
            try:
                r = requests.post(self.url, headers = self.headers, data = json.dumps(body), timeout = self.timeout)
                if r.status_code in (429, 500, 502, 503, 504, 520, 524):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if 400 <= r.status_code < 500:
                    # Bad request or credentials: retrying cannot help
                    raise PermissionError(f"HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                data = r.json()
                if "error" in data and not data.get("choices"):
                    raise RuntimeError(str(data["error"])[:300])
                choice = data["choices"][0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                reasoning = extract_reasoning(msg)
                if reasoning is None:
                    reasoning, content = split_inline_reasoning(content)
                usage = data.get("usage") or {}
                with self.lock:
                    self.usage["requests"] += 1
                    self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                    self.usage["completion_tokens"] += usage.get("completion_tokens", 0) or 0
                return {
                    "content": content,
                    "reasoning": reasoning,
                    "tool_calls": msg.get("tool_calls"),
                    "finish_reason": choice.get("finish_reason") or choice.get("native_finish_reason"),
                    "message": {k: msg[k] for k in MESSAGE_FIELDS if k in msg},
                }
            except PermissionError:
                raise
            except (requests.RequestException, RuntimeError, ValueError, KeyError) as e:
                with self.lock:
                    self.usage["retries"] += 1
                if attempt == self.retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)


def add_api_args(parser):
    parser.add_argument("--api_model", type = str, default = None, help = "Source the responses from an OpenAI-compatible chat API instead of the local model (model name or OpenRouter slug); -m then provides the tokenizer and chat template")
    parser.add_argument("--api_base", type = str, default = "https://openrouter.ai/api/v1", help = "Chat API base URL, default: OpenRouter")
    parser.add_argument("--api_key", type = str, default = None, help = "API key, default: $OPENROUTER_API_KEY, $OPENAI_API_KEY or ~/.config/openrouter_key")
    parser.add_argument("--api_concurrency", type = int, default = 8, help = "Concurrent API requests, default: 8")
    parser.add_argument("--api_temperature", type = float, default = 0.8, help = "Sampling temperature for API requests, default: 0.8")
    parser.add_argument("--api_extra", type = json.loads, default = {}, help = 'JSON merged into every request body, e.g. {"reasoning": {"effort": "high"}, "provider": {"order": ["DeepSeek"]}}')


def api_from_args(args) -> ApiChat:
    api_key = resolve_api_key(args.api_key, args.api_base)
    if api_key is None:
        print(" !! No API key (--api_key, $OPENROUTER_API_KEY, $OPENAI_API_KEY or ~/.config/openrouter_key)")
    api = ApiChat(args.api_base, api_key, args.api_model, args.api_temperature, args.max_new_tokens, args.api_extra)
    print(f" -- Responses from {args.api_model} at {args.api_base}, {args.api_concurrency} concurrent requests")
    return api


def parse_tool_calls(tool_calls: list) -> list:
    """Tool calls as the chat templates take them, with the arguments as a dict. Raises ValueError on arguments
    that aren't a JSON object, as in a response cut off inside a call"""
    parsed = []
    for tc in tool_calls:
        func = tc["function"]
        arguments = func.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError(f"tool call arguments are not an object: {str(arguments)[:80]}")
        parsed.append({"id": tc.get("id"), "type": "function",
                       "function": {"name": func["name"], "arguments": arguments}})
    return parsed


def assistant_message(content: str, reasoning: str | None, tool_calls: list | None = None) -> dict:
    """The finished turn as the chat templates expect it: reasoning under both of the field names in use"""
    m = {"role": "assistant", "content": content}
    if reasoning:
        m["reasoning_content"] = reasoning
        m["thinking"] = reasoning
    if tool_calls:
        m["tool_calls"] = parse_tool_calls(tool_calls)
    return m


def produced_end(response_text: str, assistant: dict) -> int:
    """Where the model's own output ends in the rendered turn of a response that ran into the token limit: after
    the last text it wrote. What follows is the template closing a turn the model never finished"""
    text = assistant["content"].rstrip() or (assistant.get("reasoning_content") or "").rstrip()
    tail = text[-80:]
    pos = response_text.rfind(tail) if tail else -1
    if pos < 0:
        raise ValueError("end of a truncated response not found in the rendered turn")
    return pos + len(tail)


def response_token_ids(tokenizer, messages: list, assistant: dict, template_vars: dict, warn = print,
                       truncated: bool = False):
    """
    The response token stream a local generator would have produced for `assistant`, as the difference
    between the template rendered with the generation prompt and rendered with the finished turn, tokenized
    with special tokens (think tags, end-of-turn). A template whose finished turn does not extend its own
    generation prompt falls back to the longest common prefix.

    Only what the model produced is kept. Whitespace the template puts after the end-of-turn token is dropped,
    and for a response that ran into the token limit (truncated) so is everything the template adds to close
    the unfinished turn: the closing think tag and the end-of-turn token the model never reached.
    Returns a list of ints.
    """
    prompt_text = tokenizer.hf_render_chat_template(messages, add_generation_prompt = True, **template_vars)
    full_text = tokenizer.hf_render_chat_template(messages + [assistant], add_generation_prompt = False, **template_vars)
    if full_text.startswith(prompt_text):
        response_text = full_text[len(prompt_text):]
    else:
        n = 0
        for a, b in zip(prompt_text, full_text):
            if a != b:
                break
            n += 1
        warn(f" !! Chat template's finished turn does not extend its generation prompt; "
             f"response taken from the common prefix ({len(prompt_text) - n} prompt characters dropped)")
        response_text = full_text[n:]
    response_text = response_text.rstrip()
    if truncated and not assistant.get("tool_calls"):
        response_text = response_text[:produced_end(response_text, assistant)]
    return tokenizer.encode(response_text, add_bos = False, encode_special_tokens = True)[0].tolist()
