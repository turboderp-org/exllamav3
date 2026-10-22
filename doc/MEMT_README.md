# Serving MemT-1.0-Nano with exllamav3 and TabbyAPI

*Draft release guide. Paths and numbers below are from a 2x RTX 3090 box.*

[MemT-1.0-Nano](https://huggingface.co/) is Qwen3.6-35B-A3B with a native memory block on its ten
full-attention layers. Instead of re-reading the whole conversation on every message, the model
can file earlier turns into a small latent memory (40 MiB per conversation) and answer from it.
This fork of exllamav3 runs that model, memory included, behind any OpenAI-style chat client.
TabbyAPI needs no changes.

## What you get

- Chat as usual through TabbyAPI (or any client that resends the message list).
- Type `/commit` and the conversation so far is memorized; the prompt the model actually
  reads shrinks to the system prompt plus whatever came after. Or let the server memorize on
  its own every N exchanges.
- Edits roll memory back. Regenerations don't double-memorize. Chats don't leak into each other.
- Memory can be kept on disk, so it survives a server restart.
- Decode speed with memory is the same as without. Long pasted documents commit at prefill speed.

## Requirements

- Two 24 GB GPUs (the 6 bpw EXL3 quant is ~27 GB; the memory block is bf16, ~0.9 GB).
- This fork of exllamav3 (branch with MemT support) installed into TabbyAPI's venv in editable
  mode, `flash-attn`, and `fla` (flash-linear-attention, for fast long commits; optional, falls
  back to a slow exact scan).
- The model directory: an EXL3 quant of Qwen3.6-35B-A3B plus the MemT files. See "Model
  directory" below.

## Quick start

    cd tabbyAPI
    EXL3_MEMT_SNAPSHOT_DIR=/path/to/memt_snapshots \
      ./venv/bin/python main.py --model-name /path/to/MemT-1.0-Nano-exl3-6.00bpw \
      --draft-mode disabled --max-seq-len 32768 --cache-size 32768 --gpu-split 20 23 \
      --tool-format auto --reasoning-start-token "<think>" --reasoning-end-token "</think>"

Open the Tabby chat UI and talk to it. By default every exchange is memorized as soon as the
next message arrives, so the model is always running on a short prompt. Send `/commit` at any
time to force it, or launch with `EXL3_MEMT_COMMIT_EVERY=0` to memorize only when asked.

## Using it from the chat window

| You type | What happens |
|---|---|
| `/commit` (alone) | Everything before this message is memorized. Reply: "Memorized 5 exchanges." No model call. |
| `some message` + a line `/commit` | Same, and the model answers the message. The `/commit` line is invisible to it. |
| `/commit every 3` (in a message or the system prompt) | From now on, memorize on its own every 3 exchanges. `every 0` turns that off. |
| Edit an earlier message | Memory rolls back to just before the edit. |
| Regenerate | Nothing new is memorized; the same memory is reused. |

Clients that send `template_vars` with a request can do the same without typing:
`{"memt_commit": true}` and/or `{"memt_commit_every": 3}`.

Note: if your model's chat template has thinking on by default, give replies a generous token
budget or send `{"enable_thinking": false}`, otherwise the budget goes to reasoning.

## Settings (environment variables on the launch line)

| Variable | Default | Meaning |
|---|---|---|
| `EXL3_MEMT_COMMIT_EVERY` | `1` | Memorize on its own after this many new exchanges. `0` = only on `/commit`. |
| `EXL3_MEMT_CONTEXT_TURNS` | `0` | Always keep this many of the newest exchanges as plain text too. |
| `EXL3_MEMT_COMMIT_GENERATED` | off | Memorize replies as generated (reasoning included, like the original MemT runtime) rather than as the client re-sends them. |
| `EXL3_MEMT_SNAPSHOT_DIR` | off | Directory for memory on disk (survives restarts). 40 MiB per memorized point. |
| `EXL3_MEMT_SNAPSHOT_DISK_MB` | `16384` | Disk budget for the above (least recently used evicted). |
| `EXL3_MEMT_CACHE_MB` | `2048` | RAM budget for memory states. |
| `EXL3_MEMT_VERBOSE` | off | Print one line per request: turns matched / memorized / kept raw, prompt length. |
| `EXL3_MEMT_GRAPH` | on | CUDA-graph the decode-time memory readout. |
| `EXL3_MEMT_CHUNKED` | on | Use fla's chunked kernel for commits of 128+ tokens. |

## How it works, briefly

The server is stateless: every request resends the whole message list, and the client may
edit, regenerate, or switch chats. So memory follows content, not sessions. Each turn boundary
is keyed by a hash chain over the exact token ids before it. On a request the generator finds
the longest already-memorized prefix, memorizes any turns the policy says should be (one 40 MiB
snapshot per boundary, LRU in RAM and optionally on disk), and runs the model on the system
prompt plus the raw tail plus the live message, reading that snapshot. The `/commit` lines and
the template's control turn are read and stripped before the model sees anything.

Memorizing a turn is one forward over it; the memory block's update is a gated delta rule, run
through fla's chunked KDA kernel for long turns. Reading memory during decode is additive before
`o_proj`, so the graph-captured attention block stays in use and the readout is added after it.

## Model directory

A serving directory is an ordinary EXL3 quant of the backbone plus the MemT pieces from the
release: `memt_block.safetensors` (the memory block weights), the MemT `config.json` /
`configuration.json`, and the release `chat_template.jinja` with a small addition that turns
`template_vars` into a control turn (see `doc/memt_serving_plan.md`, section 5). No requant is
needed; the backbone weights are byte-identical to the stock quant.

## Limits

- No tensor parallelism with memory.
- Batched requests work; each carries its own memory state.
- The memory block is frozen. Training it is a separate project.

## Tests

With the server stopped (they need both GPUs):

    python util/memt_gen_test.py    /path/to/model   # parity: alone / batched / cached, zero state == plain
    python util/memt_chat_test.py   /path/to/model   # recall, regenerate, edit rollback, separate chats
    python util/memt_commit_test.py /path/to/model   # every control, canned replies, generated replies, disk
