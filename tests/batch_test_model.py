import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def parse_cmdline_field(value, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(a) for a in value]
    raise ValueError(f"batch {name} must be a string or list")


def load_batch(path: Path, single) -> tuple[list[str], list[dict]]:
    with open(path, "r") as f:
        spec = yaml.safe_load(f)

    if isinstance(spec, list):
        return [], spec
    if not isinstance(spec, dict):
        raise ValueError("batch spec must be a mapping or list")

    defaults = spec.get("defaults", [])
    runs = spec.get("runs", []) if not single else spec.get("single", [])
    defaults = parse_cmdline_field(defaults, "defaults")
    if not isinstance(runs, list):
        raise ValueError("batch runs must be a list")
    return defaults, runs


def has_model_arg(args: list[str]) -> bool:
    return "-m" in args or "--model_dir" in args


def run_name(index: int, run: dict) -> str:
    name = run.get("name")
    if name:
        return name
    model_dir = run.get("model_dir")
    if model_dir:
        return Path(model_dir).name or f"run_{index:03d}"
    return f"run_{index:03d}"


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def build_command(
    python: str,
    test_script: Path,
    defaults: list[str],
    run: dict,
    name: str,
    result_file: Path,
) -> list[str]:
    args = [str(a) for a in defaults]
    args.extend(parse_cmdline_field(run.get("args"), "run args"))

    model_dir = run.get("model_dir")
    if model_dir and not has_model_arg(args):
        args = ["-m", str(model_dir)] + args

    return [
        python,
        str(test_script),
        *args,
        "--run_name",
        name,
        "--result_json",
        str(result_file),
    ]


def write_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent = 2)
        f.write("\n")


def summarize(results: list[dict]) -> dict:
    summary = {
        "total": len(results),
        "ok": 0,
        "failed": 0,
        "task_pass": {},
        "task_fail": {},
    }

    for result in results:
        if result.get("ok"):
            summary["ok"] += 1
        else:
            summary["failed"] += 1

        for task, task_result in result.get("result", {}).get("tasks", {}).items():
            key = "task_pass" if task_result.get("pass") else "task_fail"
            summary[key][task] = summary[key].get(task, 0) + 1

    return summary


def main(args) -> int:

    repo_root = Path(__file__).resolve().parents[1]
    test_script = repo_root / "tests" / "test_model.py"
    output_dir = args.output_dir
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(parents = True, exist_ok = True)
    logs_dir.mkdir(parents = True, exist_ok = True)

    defaults, runs = load_batch(args.batch, args.single)
    if not runs:
        raise ValueError("No runs found in batch spec")

    combined = []
    results_jsonl = output_dir / "results.jsonl"
    with open(results_jsonl, "w"):
        pass

    for index, run in enumerate(runs, start = 1):
        name = run_name(index, run)
        slug = safe_name(f"{index:03d}_{name}")
        result_file = results_dir / f"{slug}.json"
        log_file = logs_dir / f"{slug}.log"
        cmd = build_command(args.python, test_script, defaults, run, name, result_file)

        print(f"[{index}/{len(runs)}] {name}")
        print("  " + shlex.join(cmd))
        started = time.time()
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in run.get("env", {}).items()})

        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                cmd,
                cwd = repo_root,
                env = env,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end = "")
                log.write(line)
                log.flush()
            proc.wait()

        entry = {
            "name": name,
            "ok": proc.returncode == 0 and result_file.exists(),
            "returncode": proc.returncode,
            "elapsed_sec": time.time() - started,
            "command": cmd,
            "log": str(log_file),
            "result_file": str(result_file) if result_file.exists() else None,
            "result": None,
        }

        if result_file.exists():
            with open(result_file, "r") as f:
                entry["result"] = json.load(f)

        combined.append(entry)
        with open(results_jsonl, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        if not entry["ok"]:
            print(f"  failed, log: {log_file}")
            if not args.keep_going:
                break

    write_json(output_dir / "results.json", combined)
    write_json(output_dir / "summary.json", summarize(combined))
    print(f"Wrote {results_jsonl}")
    print(f"Wrote {output_dir / 'summary.json'}")
    return 0 if all(r["ok"] for r in combined) and len(combined) == len(runs) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("batch", type = Path, help = "YAML batch spec")
    parser.add_argument("-o", "--output_dir", type = Path, default = Path("test_model_runs"), help = "Directory for logs and combined results")
    parser.add_argument("--python", type = str, default = sys.executable, help = "Python executable to use")
    parser.add_argument("--keep_going", action = "store_true", help = "Continue after failed runs")
    parser.add_argument("-single", "--single", action = "store_true", help = "Run single test under `single` key")
    raise SystemExit(main(parser.parse_args()))
