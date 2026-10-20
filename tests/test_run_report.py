"""CPU tests for training/run_report.py -- the default local logging path.

No GPU/torch needed. Run: python tests/test_run_report.py
"""

import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "training"))
from run_report import (  # noqa: E402
    RunLogger, render_report, compare_reports, REPORT_SUBDIR,
    decode_example_docs, start_live_monitor)


def _rdir(out):
    return os.path.join(out, REPORT_SUBDIR)


def _data_of(html):
    """Extract and parse the injected DATA JSON blob from a rendered report."""
    blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    return json.loads(blob)


def _nojs_body(html):
    """The report body as a JS-disabled viewer (iOS Files/Quick Look, attachment
    previews) sees it: everything between <body>..</body> with <script> removed."""
    body = html.split("<body>", 1)[1].split("</body>", 1)[0]
    return re.sub(r"<script.*?</script>", "", body, flags=re.S)


def test_full_run_renders():
    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "unit-run", config={"lr": 1e-5, "r": 32})
        rep.log({"eval/held_out": 3.0}, step=0)
        for s in range(1, 6):
            rep.log({"train/loss": 3.0 - 0.1 * s, "train/lr": 1e-5,
                     "perf/tot_tok_s": 900 + s}, step=s)
        rep.update_summary({"best_val": 2.5, "steps_done": 5})
        rep.finish(exit_code=0)

        d = _rdir(out)
        for name in ("config.json", "metrics.jsonl", "summary.json", "report.html"):
            assert os.path.exists(os.path.join(d, name)), f"missing {name}"

        cfg = json.load(open(os.path.join(d, "config.json")))
        assert cfg["meta"]["status"] == "completed"
        assert cfg["config"]["r"] == 32
        summ = json.load(open(os.path.join(d, "summary.json")))
        assert summ["best_val"] == 2.5

        html = open(os.path.join(d, "report.html")).read()
        assert "/*__DATA__*/" not in html, "placeholder not replaced"
        # The data blob must be valid JSON with one run + the logged series.
        data = _data_of(html)
        assert len(data["runs"]) == 1
        run = data["runs"][0]
        assert "train/loss" in run["series"]
        assert len(run["series"]["train/loss"]) == 5
        assert run["series"]["eval/held_out"][0] == [0, 3.0]
    print("  full run: config/metrics/summary/report written + valid: OK")


def test_finish_is_idempotent():
    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "idem", config={})
        rep.log({"train/loss": 1.0}, step=1)
        rep.finish(exit_code=0)
        mtime = os.path.getmtime(os.path.join(_rdir(out), "report.html"))
        rep.finish(exit_code=1)  # second call must be a no-op (first caller wins)
        assert os.path.getmtime(os.path.join(_rdir(out), "report.html")) == mtime
        cfg = json.load(open(os.path.join(_rdir(out), "config.json")))
        assert cfg["meta"]["status"] == "completed"  # not overwritten to failed
    print("  finish idempotency (first caller wins): OK")


def test_nonfinite_and_torn_line_dropped():
    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "robust", config={})
        rep.log({"train/loss": 1.0}, step=1)
        rep.log({"train/loss": float("nan"), "train/grad_norm": float("inf")}, step=2)
        rep.log({"train/loss": 0.5}, step=3)
        # Simulate a hard kill leaving a torn final jsonl line (no finish()).
        with open(os.path.join(_rdir(out), "metrics.jsonl"), "a") as f:
            f.write('{"step": 4, "train/loss": 0.4')  # no newline, unterminated
        path = render_report(out)  # must render without raising
        run = _data_of(open(path).read())["runs"][0]
        steps = [p[0] for p in run["series"]["train/loss"]]
        assert steps == [1, 3], f"NaN/torn rows not dropped: {steps}"
        assert "train/grad_norm" not in run["series"], "inf value kept"
    print("  non-finite values + torn final line dropped: OK")


def test_render_from_partial_after_crash():
    # A crash before finish() leaves config.json + metrics.jsonl but no summary;
    # render must still succeed and reflect a non-completed status.
    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "crashed", config={"lr": 2e-5})
        rep.log({"train/loss": 2.0}, step=1)
        # no finish(): summary.json never written
        assert not os.path.exists(os.path.join(_rdir(out), "summary.json"))
        path = render_report(out)
        run = _data_of(open(path).read())["runs"][0]
        assert run["summary"] == {}
        assert run["meta"]["status"] == "running"  # never finished
        assert run["series"]["train/loss"] == [[1, 2.0]]
    print("  render from partial (crash before finish): OK")


def test_compare_overlays_runs():
    with tempfile.TemporaryDirectory() as root:
        outs = []
        for name, base in (("sft", 3.0), ("ebft", 3.2)):
            out = os.path.join(root, name)
            rep = RunLogger(out, name, config={"arm": name, "lr": 1e-5})
            for s in range(1, 5):
                rep.log({"train/loss": base - 0.1 * s, "eval/held_out": base - 0.05 * s}, step=s)
            rep.update_summary({"best_val": base - 0.2})
            rep.finish(exit_code=0)
            outs.append(out)

        cmp_path = os.path.join(root, "cmp.html")
        compare_reports(outs, cmp_path)
        data = _data_of(open(cmp_path).read())
        assert len(data["runs"]) == 2, "both runs must be in the compare payload"
        labels = [r["meta"]["label"] for r in data["runs"]]
        assert labels == ["sft", "ebft"]
        # Each run keeps its own series so the chart JS can overlay them.
        for r in data["runs"]:
            assert len(r["series"]["train/loss"]) == 4
        # A run dir with no metrics is skipped, not fatal.
        empty = os.path.join(root, "empty")
        os.makedirs(os.path.join(empty, REPORT_SUBDIR))
        compare_reports(outs + [empty], cmp_path)
        assert len(_data_of(open(cmp_path).read())["runs"]) == 2
    print("  compare overlays multiple runs (+ skips empty): OK")


def test_compare_dedupes_labels():
    with tempfile.TemporaryDirectory() as root:
        outs = []
        for i in range(2):
            out = os.path.join(root, f"dir{i}")
            # Same run_name on purpose -> labels must be disambiguated.
            rep = RunLogger(out, "same_name", config={})
            rep.log({"train/loss": 1.0}, step=1)
            rep.finish(exit_code=0)
            outs.append(out)
        cmp_path = os.path.join(root, "cmp.html")
        compare_reports(outs, cmp_path)
        labels = [r["meta"]["label"] for r in _data_of(open(cmp_path).read())["runs"]]
        assert len(set(labels)) == 2, f"labels not disambiguated: {labels}"
    print("  compare disambiguates duplicate labels: OK")


def test_static_prerender_single_run_no_js():
    # The report must render WITHOUT JavaScript: iOS Files/Quick Look and
    # attachment previews run the file JS-disabled, so the charts/cards/config
    # are pre-rendered to static HTML+SVG in Python. Assert they survive with
    # every <script> stripped.
    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "static-single", config={"lr": 1e-5, "model": "m/x"})
        rep.log({"eval/held_out": 3.0}, step=0)
        for s in range(1, 6):
            rep.log({"train/loss": 3.0 - 0.1 * s}, step=s)
        rep.update_summary({"best_val": 2.5, "steps_done": 5})
        rep.finish(exit_code=0)

        html = open(os.path.join(_rdir(out), "report.html")).read()
        assert "__STATIC_" not in html, "a static placeholder token was left unreplaced"
        body = _nojs_body(html)
        assert "static-single" in body, "title missing from no-JS body"
        assert body.count("<svg") >= 2, "charts not pre-rendered (no <svg> without JS)"
        assert 'class="plot"' in body, "no plotted series path in static SVG"
        assert 'class="card"' in body, "summary cards not pre-rendered"
        assert "best_val" in body and "table class=\"cfg\"" in body, "summary/config missing"
    print("  static pre-render (single run, no JS): OK")


def test_static_prerender_compare_no_js():
    with tempfile.TemporaryDirectory() as root:
        outs = []
        for name, base in (("sft", 3.0), ("ebft", 3.2)):
            out = os.path.join(root, name)
            rep = RunLogger(out, name, config={"arm": name, "lr": 1e-5})
            for s in range(1, 5):
                rep.log({"train/loss": base - 0.1 * s}, step=s)
            rep.update_summary({"best_val": base - 0.2})
            rep.finish(exit_code=0)
            outs.append(out)
        cmp_path = os.path.join(root, "cmp.html")
        compare_reports(outs, cmp_path)

        html = open(cmp_path).read()
        assert "__STATIC_" not in html, "a static placeholder token was left unreplaced"
        body = _nojs_body(html)
        assert "Comparison" in body, "comparison title missing from no-JS body"
        assert body.count('class="chip"') == 2, "legend chips not pre-rendered per run"
        assert body.count('class="cmp"') == 2, "summary + config compare tables missing"
        assert 'class="diff"' in body, "differing-config rows not highlighted statically"
        # Overlaid runs -> a chart SVG carrying more than one plotted series.
        assert body.count('class="plot"') > body.count("<svg"), "series not overlaid in static charts"
    print("  static pre-render (compare, no JS): OK")


class _FakeTok:
    """Stands in for exllamav3's Tokenizer: decode(1-D tensor) -> str."""
    def decode(self, t, decode_special_tokens=False):
        return " ".join(f"t{i}" for i in t.tolist())


def test_decode_example_docs():
    # Requires torch (CPU only); the rest of this file stays torch-free.
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  decode_example_docs: SKIPPED (no torch)")
        return
    tok = _FakeTok()
    # plain SFT example: masked prompt prefix + supervised response
    docs = decode_example_docs(tok, {"input_ids": [5, 6, 7, 8],
                                     "labels": [-100, -100, 7, 8]})
    assert docs == [{"segments": [{"text": "t5 t6", "sup": False, "n": 2},
                                  {"text": "t7 t8", "sup": True, "n": 2}],
                     "n_prompt": 2, "n_sup": 2}]
    # packed block: two docs split on seg_ids; trailing pads (label -100 after
    # the response, inheriting the last doc's seg id) must land in neither span
    docs = decode_example_docs(tok, {
        "input_ids": [1, 2, 3, 4, 5, 6, 0, 0],
        "labels": [-100, 2, 3, -100, 5, 6, -100, -100],
        "seg_ids": [0, 0, 0, 1, 1, 1, 1, 1]})
    assert len(docs) == 2
    assert docs[0] == {"segments": [{"text": "t1", "sup": False, "n": 1},
                                    {"text": "t2 t3", "sup": True, "n": 2}],
                       "n_prompt": 1, "n_sup": 2}
    assert docs[1] == {"segments": [{"text": "t4", "sup": False, "n": 1},
                                    {"text": "t5 t6", "sup": True, "n": 2}],
                       "n_prompt": 1, "n_sup": 2}
    print("  decode_example_docs (plain + packed + pad trim): OK")


def test_live_monitor_endpoints():
    import urllib.error
    import urllib.request

    def get(url):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "live-run", config={"lr": 1e-5})
        rep.log({"train/loss": 2.5}, step=1)

        def batch_fn(s):
            if not (0 < s <= 10):
                raise ValueError("step must be in [1, 10]")
            return {"step": s, "micro_batches": [
                {"micro": 0, "epoch": 0, "sequences": [
                    {"index": 0, "docs": [{"prompt": "p", "response": "r",
                                           "n_prompt": 1, "n_sup": 1}]}]}]}

        srv = start_live_monitor(out, batch_fn=batch_fn, port=0,
                                 open_browser=False,
                                 live_info={"total_steps": 10, "first_step": 1,
                                            "run_name": "live-run"})
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            # / -> LIVE config injected (static template default is null)
            code, html = get(base + "/")
            assert code == 200
            assert '"total_steps":10' in html, "LIVE info not injected"
            assert "/*__LIVE__*/null" not in html, "LIVE left off in live page"
            assert "__STATIC_TITLE__" not in html, "static tokens not replaced"
            # /metrics -> the streamed jsonl
            code, body = get(base + "/metrics")
            assert code == 200 and '"train/loss": 2.5' in body
            # /batch -> provider JSON; out-of-range / garbage -> 400
            code, body = get(base + "/batch?step=3")
            assert code == 200 and json.loads(body)["step"] == 3
            assert get(base + "/batch?step=99")[0] == 400
            assert get(base + "/batch?step=x")[0] == 400
            assert get(base + "/nope")[0] == 404
        finally:
            srv.shutdown()
            srv.server_close()
        # the shareable artifact must stay non-live and dataset-free
        rep.finish(0)
        with open(os.path.join(_rdir(out), "report.html")) as f:
            static = f.read()
        assert "/*__LIVE__*/null" in static, "static report must keep LIVE off"
        assert "micro_batches" not in _data_of(static), "dataset leaked into report"
    print("  live monitor endpoints + static stays non-live: OK")


if __name__ == "__main__":
    test_full_run_renders()
    test_finish_is_idempotent()
    test_nonfinite_and_torn_line_dropped()
    test_render_from_partial_after_crash()
    test_compare_overlays_runs()
    test_compare_dedupes_labels()
    test_static_prerender_single_run_no_js()
    test_static_prerender_compare_no_js()
    test_decode_example_docs()
    test_live_monitor_endpoints()
    print("ALL RUN_REPORT TESTS PASSED")
