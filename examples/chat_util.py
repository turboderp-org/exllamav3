from __future__ import annotations
import re
import pyperclip
import json
import random
from exllamav3.util.file import disk_lru_cache

def copy_last_codeblock(text: str, num) -> str | None:
    pattern = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    if num > len(matches):
        num = len(matches)
    snippet = matches[-num].strip()
    pyperclip.copy(snippet)
    return snippet

def extract_svg(s: str, begin: str = "<svg", end: str = "</svg>"):

    # Find all tag occurrences in order
    pattern = re.compile(rf"{re.escape(begin)}|{re.escape(end)}")
    tags = list(pattern.finditer(s))

    best = None
    for i in range(len(tags) - 1):
        t1, t2 = tags[i], tags[i+1]
        if t1.group() == begin and t2.group() == end:
            start = t1.start()
            stop  = t2.end()
            length = stop - start
            if best is None or length > best[0]:
                best = (length, start, stop)

    if not best:
        return None

    _, start, stop = best
    return s[start:stop]


def _fetch_github_json(url: str) -> list[dict]:
    import urllib.request
    req = urllib.request.Request(url, headers = {"User-Agent": "benchmark-sampler"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


@disk_lru_cache("_load_truthfulqa")
def _load_truthfulqa() -> list[str]:
    """817 questions spanning health, law, finance, politics, etc."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split = "validation")
    return [row["question"] for row in ds]


@disk_lru_cache("_load_simpleqa")
def _load_simpleqa() -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("basicv8vc/SimpleQA", split = "test")
    return [row["problem"] for row in ds]


@disk_lru_cache("_load_arc_challenge")
def _load_arc_challenge() -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split = "test")
    return [row["question"] for row in ds]


@disk_lru_cache("_load_commonsenseqa")
def _load_commonsenseqa() -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split = "validation")
    return [row["question"] for row in ds]


@disk_lru_cache("_load_bullshitbench_v2_")
def _load_bullshitbench_v2() -> list[str]:
    url = "https://raw.githubusercontent.com/petergpt/bullshit-benchmark/main/questions.v2.json"
    data = _fetch_github_json(url)
    questions = []
    for t in data["techniques"]:
        questions += [q["question"] for q in t.get("questions", data) if "question" in q]
    return questions


@disk_lru_cache("_load_mt_bench")
def _load_mt_bench() -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split = "train")
    return [row["prompt"][0] for row in ds]


@disk_lru_cache("_load_wildbench")
def _load_wildbench() -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("allenai/WildBench", "v2", split = "test")
    questions = []
    for row in ds:
        turns = row["conversation_input"]
        for turn in turns:
            if turn["role"] == "user":
                if turn.get("toxic") or turn.get("redacted"):
                    break
                text = turn["content"].strip()
                if text:
                    questions.append(text)
                break  # only first user turn
    return questions


bench_sources = {
    "truthfulqa": _load_truthfulqa,
    "simpleqa": _load_simpleqa,
    "commonsenseqa": _load_commonsenseqa,
    "bullshitbench": _load_bullshitbench_v2,
    "mtbench": _load_mt_bench,
    "wildbench": _load_wildbench,
}


def get_sample_sources():
    return bench_sources.keys()


def sample_question(source: str):
    if source not in bench_sources:
        return None
    questions = bench_sources[source]()
    return random.choice(questions)