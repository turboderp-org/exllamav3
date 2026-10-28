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

    def complete(self, messages: list, seed: int | None = None) -> dict:
        """Returns {"content", "reasoning", "finish_reason"}; raises after the last retry"""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed
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
                    "finish_reason": choice.get("finish_reason") or choice.get("native_finish_reason"),
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


def assistant_message(content: str, reasoning: str | None) -> dict:
    """The finished turn as the chat templates expect it: reasoning under both of the field names in use"""
    m = {"role": "assistant", "content": content}
    if reasoning:
        m["reasoning_content"] = reasoning
        m["thinking"] = reasoning
    return m


def response_token_ids(tokenizer, messages: list, assistant: dict, template_vars: dict, warn = print):
    """
    The response token stream a local generator would have produced for `assistant`, as the difference
    between the template rendered with the generation prompt and rendered with the finished turn, tokenized
    with special tokens (think tags, end-of-turn). A template whose finished turn does not extend its own
    generation prompt falls back to the longest common prefix.
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
    return tokenizer.encode(response_text, add_bos = False, encode_special_tokens = True)[0].tolist()
