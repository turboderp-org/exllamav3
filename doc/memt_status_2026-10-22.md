# MemT-1.0-Nano serving: where it stands (2026-10-22)

Written for you to refresh and try it yourself. Plain words first, details after.
Everything here is on branch `claude/brave-lovelace-myoqvj`, pushed, with tests passing
after every step. The design notes and the long version live in `doc/memt_serving_plan.md`.

## What you have

A chat server (TabbyAPI, your normal UI) running MemT-1.0-Nano, which is Qwen3.6-35B-A3B plus a
memory block. Instead of re-reading the whole conversation every message, the model can file
earlier exchanges away into a small internal memory and answer from that. From the chair:

- You chat normally. Nothing special happens until you ask for it (with the launch line below).
- Put `/commit` on a line of its own in a message, or send `/commit` by itself, and everything
  before that message is memorized. A bare `/commit` answers "Memorized 5 exchanges." and costs
  nothing. After that, the model sees only the system prompt, anything newer than the commit, and
  your current message, and still knows what came before.
- `/commit every 3` (in a message, or in the system prompt) makes it memorize on its own every
  three exchanges, no typing.
- Editing an old message rolls memory back to that point. Regenerating doesn't memorize twice.
  Different chats share nothing.
- Memory survives restarting the server (it is written to `/mnt/two/memt_snapshots`).
- Replies run at full speed with memory in use (73 tok/s vs 79 without; it was 51 this morning).
- Pasting a long document and committing it takes as long as reading it would (7.6k tokens in
  1.5 s), and the facts in it are recalled.

## Launch

From `~/exl3/tabbyAPI` (your `config.yml` is untouched; everything is flag overrides):

    EXL3_MEMT_VERBOSE=1 EXL3_MEMT_COMMIT_EVERY=0 EXL3_MEMT_SNAPSHOT_DIR=/mnt/two/memt_snapshots \
      ./venv/bin/python main.py --model-name /mnt/two/weights/MemT-1.0-Nano-exl3-6.00bpw \
      --draft-mode disabled --max-seq-len 32768 --cache-size 32768 --gpu-split 20 23 --port 5055 \
      --tool-format auto --muse-glimmer false --reasoning-start-token "<think>" --reasoning-end-token "</think>"

Then open http://127.0.0.1:5055/ui/ (your browser has the key). Loads in about 12 s.

To stop it: `kill $(pgrep -f '^\./venv/bin/python main\.py')`, and wait for the VRAM to drop
before loading anything else.

To watch what memory is doing, in another terminal:

    tail -f <the log you redirected to> | grep --line-buffered -- "-- memt:"

One line per request: how many turns matched, committed, kept raw, and how short the prompt became.

## The knobs (all environment variables on the launch line)

| Variable | Default | What it does |
|---|---|---|
| `EXL3_MEMT_COMMIT_EVERY` | 1 | Memorize on its own after this many new exchanges. 1 = every message (memory always full). 0 = only when asked with `/commit`. Overridden per chat by `/commit every N`. |
| `EXL3_MEMT_CONTEXT_TURNS` | 0 | Keep this many of the newest exchanges as plain text even when memorized. |
| `EXL3_MEMT_COMMIT_GENERATED` | off | Memorize replies as the model actually wrote them (thinking included) instead of the copy the UI sends back (thinking stripped). The original MemT does the former. Which remembers better is untested; that is the experiment to run. |
| `EXL3_MEMT_SNAPSHOT_DIR` | off | Where to keep memory on disk so it survives restarts. 40 MiB per memorized point. |
| `EXL3_MEMT_SNAPSHOT_DISK_MB` | 16384 | Disk budget for the above, oldest-unused evicted first. |
| `EXL3_MEMT_CACHE_MB` | 2048 | RAM budget for memory states (about 50 of them). |
| `EXL3_MEMT_VERBOSE` | off | The one-line-per-request log. |
| `EXL3_MEMT_GRAPH` | on | CUDA-graph the memory readout during decode (the speed fix). `0` disables. |
| `EXL3_MEMT_CHUNKED` | on | Fast chunked kernel for long commits. `0` forces the slow exact reference. |

Clients that can send `template_vars` can also trigger a commit per request with
`{"memt_commit": true}` or set the pace with `{"memt_commit_every": 3}`; Tabby's
`template_vars_default` can set those per model. This works through a small addition to the
sidecar's `chat_template.jinja` (original kept as `.orig`).

## Things to know

- Thinking is on by default in your UI. Either turn it off or give replies a generous budget, or
  the whole budget goes to thinking and the visible answer is empty.
- A message that is only `/commit` never reaches the model. Only `/commit` and `/commit every N`
  are commands; anything else (e.g. `/remember`) is plain text to the model.
- The first long commit after the model loads is fast now, but the very first Tabby start after
  a code change can take a minute longer while a kernel compiles. Subsequent starts are 12 s.
- Memory is matched on the exact text the UI sends back, so changing the system prompt starts a
  fresh memory for that chat.

## What was built today, in order

1. Commit policy and the `/commit` controls, plus the request-field route. (c694a7d5)
2. Bare `/commit` gets a fixed reply and is kept out of memory. (d7fa47e2)
3. Memorize replies as generated, switchable. (4dfed0ed)
4. Memory on disk, survives restart. (654356a8)
5. Full decode speed with memory: graph path plus a CUDA-graphed readout. (306d74ed)
6. Long documents through fla's chunked KDA kernel. (414ab691)

Tests, run with `~/exl3/qlora-venv/bin/python <script> /mnt/two/weights/MemT-1.0-Nano-exl3-6.00bpw`
(stop Tabby first, they need both cards):

- `util/memt_commit_test.py` (46 checks: every control, canned replies, generated replies, disk)
- `util/memt_chat_test.py` (default policy: recall, regenerate, edit rollback, separate chats)
- `util/memt_gen_test.py` (alone / batched / cached parity, zero state equals plain model)

## Not done

- Tensor parallelism with memory.
- The generated-vs-rendered recall comparison (run the same chat both ways).
- Training the memory block (`doc/memt_plan.md` phase 2). That is a research project, not a
  feature for someone using the model.
