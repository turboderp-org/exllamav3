#!/usr/bin/env python3
"""
Gateway regression smoke test for OpenAI-compatible streaming endpoint.

Designed for local serving checks (e.g. tabby/OpenAI API proxy) without starting
another server process. Uses curl + SSE parsing, matching the user's manual flow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass


@dataclass
class Case:
    name: str
    user_prompt: str
    expect_hangul: bool = False
    expect_latin: bool = False
    max_tokens: int = 192


SYSTEM_PROMPT_DEFAULT = (
    "당신은 친절한 AI입니다. 생각과 응답 모두 입력된 언어와 동일한 언어로 작성하세요."
)


CASES: list[Case] = [
    Case(
        name="en_basic",
        user_prompt="Hello! Reply in English in one or two short sentences.",
        expect_latin=True,
    ),
    Case(
        name="ko_basic",
        user_prompt="안녕하세요! 한국어로만 짧게 인사해 주세요.",
        expect_hangul=True,
    ),
    Case(
        name="ko_math",
        user_prompt="54*33 풀이와 답을 한국어로 작성하고 마지막 줄에는 답만 적어주세요.",
        expect_hangul=True,
    ),
]


def _build_payload(model: str, system_prompt: str, case: Case, temperature: float) -> str:
    payload = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case.user_prompt},
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _looks_gibberish(text: str) -> bool:
    t = text.lower()
    if "\ufffd" in text:
        return True
    indicators = ["-sup", "viste", "nodoc", "ellini"]
    hits = sum(t.count(k) for k in indicators)
    return hits >= 6


def _run_stream(endpoint: str, api_key: str, payload: str, timeout_sec: float):
    cmd = [
        "curl",
        "-sN",
        endpoint,
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {api_key}",
        "-d",
        payload,
    ]

    started = time.perf_counter()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    timed_out = False
    out = ""
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        out = cp.stdout or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        if isinstance(e.stdout, bytes):
            out = e.stdout.decode("utf-8", errors="replace")
        else:
            out = e.stdout or ""

    elapsed = time.perf_counter() - started
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue

        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict) and obj.get("usage"):
            usage = obj.get("usage")

        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        c = delta.get("content")
        r = delta.get("reasoning")
        if isinstance(c, str) and c:
            content_parts.append(c)
        if isinstance(r, str) and r:
            reasoning_parts.append(r)

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    completion_tokens = None
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")

    return {
        "timeout": timed_out,
        "elapsed_sec": elapsed,
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "completion_tokens": completion_tokens,
    }


def _validate(case: Case, text: str) -> list[str]:
    issues: list[str] = []
    if not text.strip():
        issues.append("empty_output")
    if case.expect_hangul and re.search(r"[가-힣]", text) is None:
        issues.append("missing_hangul")
    if case.expect_latin and re.search(r"[A-Za-z]", text) is None:
        issues.append("missing_latin")
    if _looks_gibberish(text):
        issues.append("gibberish_pattern")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8088/v1/chat/completions")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default="default")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout_sec", type=float, default=120.0)
    ap.add_argument("--min_gen_tps", type=float, default=0.0)
    ap.add_argument("--system_prompt", default=SYSTEM_PROMPT_DEFAULT)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api_key (or OPENAI_API_KEY env)")

    any_fail = False
    print(f"Endpoint: {args.endpoint}")
    print(f"Model:    {args.model}")
    print("")

    for case in CASES:
        payload = _build_payload(args.model, args.system_prompt, case, args.temperature)
        result = _run_stream(args.endpoint, args.api_key, payload, args.timeout_sec)

        text = result["content"] if result["content"] else result["reasoning"]
        issues = _validate(case, text)
        if result["timeout"]:
            issues.append("timeout")

        gen_tps = None
        ct = result["completion_tokens"]
        if isinstance(ct, int) and ct > 0 and result["elapsed_sec"] > 0:
            gen_tps = ct / result["elapsed_sec"]
            if args.min_gen_tps > 0 and gen_tps < args.min_gen_tps:
                issues.append(f"slow_gen_tps<{args.min_gen_tps}")

        status = "PASS" if not issues else "FAIL"
        if issues:
            any_fail = True

        print(f"[{case.name}] {status}")
        print(f"  elapsed: {result['elapsed_sec']:.2f}s")
        if gen_tps is not None:
            print(f"  gen_tps: {gen_tps:.2f}")
        print(f"  issues:  {issues}")
        preview = text.replace("\n", " ")[:280]
        print(f"  preview: {preview}")
        print("")

    if any_fail:
        raise SystemExit(2)
    print("All gateway regression smoke checks passed.")


if __name__ == "__main__":
    main()
