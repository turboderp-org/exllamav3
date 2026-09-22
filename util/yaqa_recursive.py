"""Collect YAQA factors with one decoder block's factors resident at a time.

Recursive scheduling follows section 5 of "BaKron: Efficient Quantization with Kronecker-Factored Hessians", Johann Birnick and Rayan Saab (2026), https://arxiv.org/abs/2608.06291. The adaptation here uses that recursive boundary strategy with the existing YAQA factor updates; disk-backed storage and resumable per-tensor publication are implementation choices, not a change to the estimator.

An initial model pass records each row's range input, decoder call metadata and sampled Fisher gradients at the range output. Recursive splits replay these boundaries to derive midpoint inputs and gradients; leaves run the existing Collector schedule without changing its factor math.

Boundary datasets live in temporary disk directories. Depth-first traversal keeps only the current recursion path's datasets, while tensor call metadata is moved to the execution device one block at a time. Scratch data is private to this invocation and is rebuilt on resume; only complete factor outputs are reusable.
"""

import argparse
import os
import re
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import PreTrainedModel
from util.yaqa_hessians import Collector, skip_names

# Replay arguments excluding hidden_states, keyword arguments, and whether hidden_states was positional.
Call = tuple[tuple[object, ...], dict[str, object], bool]
# Original and copied tensors keyed by object identity; retaining originals prevents id reuse within a row.
Copies = dict[int, tuple[torch.Tensor, torch.Tensor]]


def hidden(output: object) -> torch.Tensor:
    """Extract hidden states from a decoder result, rejecting unsupported output layouts."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Decoder output must be a tensor or a tuple beginning with one")


def move_tree(value: object, device: torch.device | str, memo: Copies) -> object:
    """Detach and move tensors inside replayable positional or keyword metadata.

    The shared memo preserves tensor aliases across argument containers and retains original tensors so their identities cannot be reused during capture. Only tensors, plain containers and scalar values are accepted: mutable cache objects cannot be replayed safely.
    """
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value, value.detach().to(device)
        return memo[id(value)][1]
    if isinstance(value, dict):
        return {k: move_tree(v, device, memo) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(v, device, memo) for v in value)
    if isinstance(value, list):
        return [move_tree(v, device, memo) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Recursive collection cannot replay call metadata of type {type(value).__name__}"
    )


def execution_device(block: nn.Module) -> torch.device:
    """Find where a block executes, rather than where its offloaded weights are stored.

    Prefer Accelerate's execution hook because offloaded parameters may be on CPU or the meta device. Unhooked blocks use their first parameter's device.
    """
    for module in block.modules():
        hook = getattr(module, "_hf_hook", None)
        device = getattr(hook, "execution_device", None)
        if device is not None:
            return torch.device(device)
    return next(block.parameters()).device


def collect_recursive(
    args: argparse.Namespace,
    model: PreTrainedModel,
    rows: list[torch.Tensor],
    targets: dict[str, tuple[str, nn.Linear]],
    first: nn.Module,
    last: nn.Module,
    schedule: list[str],
    hess_devices: list[torch.device] | None,
) -> None:
    """Collect pending targets in the inclusive first-to-last decoder range.

    Targets must already exclude validated complete outputs. Leading and trailing completed blocks are trimmed, but the full model still computes prefix activations and the sampled Fisher targets. Per-row gradients at the selected range's output are then reused for every split and leaf, preserving the unrecursed sampling and update schedule.

    Each split derives an activation and gradient boundary at its midpoint. Its children receive opposite pairs of outer and midpoint boundaries; only leaves allocate decoder factors. A requested head is collected during boundary preparation and stores Hin only. Scratch directories and hooks are released on normal return or handled failure; completed factor files remain available for a later resume.
    """
    blocks = [
        m
        for n, m in model.named_modules()
        if re.search(r"(^|\.)layers\.\d+$", n) and not any(s in n for s in skip_names)
    ]
    offset = blocks.index(first)
    blocks = blocks[offset : blocks.index(last) + 1]
    pending = {module for key, module in targets.values() if key != "lm_head"}
    active = [
        i for i, block in enumerate(blocks) if pending.intersection(block.modules())
    ]
    if active:
        offset += active[0]
        blocks = blocks[active[0] : active[-1] + 1]
        first, last = blocks[0], blocks[-1]
    print(f" -- Recomputing calibration inputs at layer {offset}", flush=True)
    devices = [execution_device(block) for block in blocks]
    embedding_weight = model.get_input_embeddings().weight
    if not isinstance(embedding_weight, torch.Tensor):
        raise TypeError("Input embedding weight must be a tensor")
    input_device = embedding_weight.device
    counts = {"forward": 0, "backward": 0}
    started = time.monotonic()
    os.makedirs(args.recursive, exist_ok=True)
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    def progress(label: str, row: int, start: float) -> None:
        """Report row progress without synchronizing every iteration."""
        if (row + 1) % args.log_interval == 0 or row + 1 == len(rows):
            elapsed = time.monotonic() - start
            eta = elapsed / (row + 1) * (len(rows) - row - 1)
            print(
                f" -- {label}: row {row + 1}/{len(rows)}, {elapsed:.1f} s, ETA {eta:.1f} s",
                flush=True,
            )

    def finish(label: str, start: float) -> None:
        """Wait for selected devices before recording completed phase time."""
        for device in set(devices):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        print(f" -- {label}: {elapsed:.3f} s", flush=True)

    def run(
        begin: int, end: int, x: torch.Tensor, calls: dict[int, Call]
    ) -> torch.Tensor:
        """Replay a half-open block range with this row's captured call metadata."""
        for index in range(begin, end):
            positional, keywords, positional_hidden = calls[index]
            copies: Copies = {}
            positional = tuple(move_tree(v, devices[index], copies) for v in positional)
            keywords = {
                k: move_tree(v, devices[index], copies) for k, v in keywords.items()
            }
            x = x.to(devices[index]).contiguous()
            if positional_hidden:
                positional = (x,) + positional
            else:
                keywords["hidden_states"] = x
            x = hidden(blocks[index](*positional, **keywords))
            counts["forward"] += 1
        return x

    def save_boundary(
        directory: Path, row: int, tensors: dict[str, torch.Tensor]
    ) -> None:
        """Spill detached contiguous tensors so scratch files retain no autograd graph."""
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
            str(directory / f"{row}.safetensors"),
        )

    def load_boundary(directory: Path, row: int) -> dict[str, torch.Tensor]:
        """Load one row boundary on CPU; missing or malformed scratch files fail the run."""
        return load_file(str(directory / f"{row}.safetensors"))

    with tempfile.TemporaryDirectory(prefix="yaqa-", dir=args.recursive) as temporary:
        root = Path(temporary)
        inputs = root / "inputs"
        gradients = root / "gradients"
        metadata = root / "calls"
        for directory in (inputs, gradients, metadata):
            directory.mkdir()
        calls: dict[int, Call] = {}
        captured: dict[str, torch.Tensor] = {}
        memo: Copies = {}

        def capture(
            index: int, positional: tuple[object, ...], keywords: dict[str, object]
        ) -> None:
            """Capture replay arguments and the range input without retaining decoder graphs."""
            if positional:
                x, positional = positional[0], positional[1:]
                positional_hidden = True
            else:
                keywords = dict(keywords)
                x = keywords.pop("hidden_states")
                positional_hidden = False
            if not isinstance(x, torch.Tensor):
                raise TypeError("Decoder hidden_states must be a tensor")
            if index == 0:
                captured["x"] = x.detach().cpu()
            calls[index] = (
                tuple(move_tree(v, "cpu", memo) for v in positional),
                {k: move_tree(v, "cpu", memo) for k, v in keywords.items()},
                positional_hidden,
            )

        def cut_graph(
            module: nn.Module,
            positional: tuple[object, ...],
            output: torch.Tensor | tuple[object, ...],
        ) -> torch.Tensor | tuple[object, ...]:
            """Make the range output a leaf so Fisher backprop stops at this boundary."""
            boundary = hidden(output).detach().requires_grad_(True)
            captured["end"] = boundary
            return (
                boundary
                if isinstance(output, torch.Tensor)
                else (boundary,) + output[1:]
            )

        head_collectors = [
            Collector(n, k, m, (hess_devices or [devices[-1]])[0], True, False)
            for n, (k, m) in targets.items()
            if k == "lm_head"
        ]
        hooks = [
            block.register_forward_pre_hook(
                lambda m, a, kw, index=index: capture(index, a, kw), with_kwargs=True
            )
            for index, block in enumerate(blocks)
        ]
        hooks.append(last.register_forward_hook(cut_graph))
        for collector in head_collectors:
            collector.begin_pass("fwd")
            hooks.append(
                targets[collector.name][1].register_forward_hook(
                    lambda m, a, out, c=collector: c.forward("fwd", a[0], None)
                )
            )
        start = time.monotonic()
        try:
            for row, ids in enumerate(rows):
                with torch.enable_grad():
                    logits = model(
                        input_ids=ids.to(input_device), use_cache=False
                    ).logits[0]
                counts["forward"] += len(blocks)
                probs = torch.softmax(
                    logits.detach().float() / args.temperature, dim=-1
                )
                # Match the unrecursed row seed; resampling at leaves would change the Fisher factors.
                generator = torch.Generator(device=probs.device).manual_seed(
                    args.seed + row
                )
                samples = torch.multinomial(
                    probs, args.samples, replacement=True, generator=generator
                )
                positions = torch.arange(probs.shape[0], device=probs.device)
                boundary_gradients = {}
                for sample in range(args.samples):
                    grad = probs.to(logits.dtype, copy=True)
                    grad[positions, samples[:, sample]] -= 1.0
                    boundary_gradients[str(sample)] = torch.autograd.grad(
                        logits,
                        captured["end"],
                        grad,
                        retain_graph=sample < args.samples - 1,
                    )[0].cpu()
                save_boundary(inputs, row, {"x": captured["x"]})
                save_boundary(gradients, row, boundary_gradients)
                torch.save(calls, metadata / f"{row}.pt")
                calls.clear()
                captured.clear()
                memo.clear()
                del logits, probs, grad, boundary_gradients
                progress("Root boundaries", row, start)
        finally:
            for hook in hooks:
                hook.remove()
        finish("Root boundaries", start)
        for collector in head_collectors:
            collector.end_pass("fwd")
            if not args.dry_run:
                collector.save_atomic(args.out_dir, args.sides)
        if head_collectors:
            del collector
        head_collectors.clear()

        def leaf(index: int, x_directory: Path, d_directory: Path) -> None:
            """Run every factor-update pass for one block, then publish its pending outputs."""
            members = set(blocks[index].modules())
            selected = [(n, k, m) for n, (k, m) in targets.items() if m in members]
            if not selected:
                return
            collectors = []
            load = {device: 0 for device in hess_devices or [devices[index]]}
            for name, key, module in sorted(
                selected,
                key=lambda t: -(t[2].in_features ** 2 + t[2].out_features ** 2),
            ):
                device = min(load, key=load.__getitem__)
                collectors.append(
                    Collector(name, key, module, device, schedule[0] == "fwd", True)
                )
                load[device] += (
                    module.in_features**2 * (schedule[0] == "fwd")
                    + module.out_features**2
                )
            kind = [schedule[0]]
            last_input = [None, None]

            def forward(
                collector: Collector,
                positional: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                """Share Hin for sibling projections and attach gradient collection to their outputs."""
                x = positional[0]
                # Hold the previous input itself: pointer equality alone is unsafe after allocator reuse.
                previous, leader = last_input
                same = previous is not None and (
                    x is previous
                    or (
                        x.data_ptr() == previous.data_ptr()
                        and x.shape == previous.shape
                        and x.stride() == previous.stride()
                        and x.dtype == previous.dtype
                        and x.device == previous.device
                    )
                )
                if not same:
                    last_input[:] = x, collector
                    leader = None
                collector.forward(kind[0], x, leader)
                if kind[0] != "fwd" and output.requires_grad:
                    output.register_hook(lambda d: collector.backward(kind[0], d))

            hooks = [
                targets[c.name][1].register_forward_hook(
                    lambda m, a, out, c=c: forward(c, a, out)
                )
                for c in collectors
            ]
            try:
                for pass_index, kind[0] in enumerate(schedule):
                    start = time.monotonic()
                    for collector in collectors:
                        collector.begin_pass(kind[0])
                    for row in range(len(rows)):
                        row_calls = torch.load(
                            metadata / f"{row}.pt", weights_only=True
                        )
                        x = load_boundary(x_directory, row)["x"].to(devices[index])
                        if kind[0] == "fwd":
                            with torch.no_grad():
                                run(index, index + 1, x, row_calls)
                        else:
                            gradients = load_boundary(d_directory, row)
                            x.requires_grad_(True)
                            output = run(index, index + 1, x, row_calls)
                            for sample in range(args.samples):
                                output.backward(
                                    gradients[str(sample)].to(output.device),
                                    retain_graph=sample < args.samples - 1,
                                )
                                counts["backward"] += 1
                            del output, gradients
                        for collector in collectors:
                            collector.end_row()
                        last_input[:] = None, None
                        del x, row_calls
                        progress(
                            f"Layer {offset + index}, pass {pass_index + 1} ({kind[0]})",
                            row,
                            start,
                        )
                    for collector in collectors:
                        collector.end_pass(kind[0])
                    finish(
                        f"Layer {offset + index}, pass {pass_index + 1} ({kind[0]})",
                        start,
                    )
                if not args.dry_run:
                    for collector in collectors:
                        collector.save_atomic(args.out_dir, args.sides)
            finally:
                for hook in hooks:
                    hook.remove()

        def split(
            begin: int, end: int, x_directory: Path, d_directory: Path, depth: int
        ) -> None:
            """Derive midpoint boundaries, then visit both children before deleting them.

            The left child uses the parent input and midpoint gradient; the right child uses the midpoint input and parent output gradient. Completed subtrees need no factor work.
            """
            if not any(
                pending.intersection(block.modules()) for block in blocks[begin:end]
            ):
                return
            if end - begin == 1:
                leaf(begin, x_directory, d_directory)
                return
            midpoint = (begin + end) // 2
            with tempfile.TemporaryDirectory(
                prefix=f"split-{begin}-{end}-", dir=root
            ) as split_temporary:
                mid_inputs = Path(split_temporary) / "inputs"
                mid_gradients = Path(split_temporary) / "gradients"
                mid_inputs.mkdir()
                mid_gradients.mkdir()
                start = time.monotonic()
                for row in range(len(rows)):
                    row_calls = torch.load(metadata / f"{row}.pt", weights_only=True)
                    x = load_boundary(x_directory, row)["x"]
                    with torch.no_grad():
                        middle = run(begin, midpoint, x, row_calls)
                    middle = middle.detach().requires_grad_(True)
                    output = run(midpoint, end, middle, row_calls)
                    gradients = load_boundary(d_directory, row)
                    result = {}
                    for sample in range(args.samples):
                        result[str(sample)] = torch.autograd.grad(
                            output,
                            middle,
                            gradients[str(sample)].to(output.device),
                            retain_graph=sample < args.samples - 1,
                        )[0]
                        counts["backward"] += end - midpoint
                    save_boundary(mid_inputs, row, {"x": middle})
                    save_boundary(mid_gradients, row, result)
                    del x, middle, output, gradients, result, row_calls
                    progress(
                        f"Split {offset + begin}:{offset + end}, depth {depth}",
                        row,
                        start,
                    )
                finish(f"Split {offset + begin}:{offset + end}, depth {depth}", start)
                split(begin, midpoint, x_directory, mid_gradients, depth + 1)
                split(midpoint, end, mid_inputs, d_directory, depth + 1)

        split(0, len(blocks), inputs, gradients, 0)

    # Counts cover selected blocks only, not prefix/suffix work or checkpoint's internal replay.
    denominator = len(blocks) * len(rows)
    print(
        f" -- Decoder-equivalent passes: forward {counts['forward'] / denominator:.3f}, "
        f"backward {counts['backward'] / denominator:.3f}; checkpoint recomputation excluded from forward count"
    )
    print(f" -- Recursive total: {time.monotonic() - started:.3f} s")
    used = set(devices) | set(hess_devices or [])
    print(
        " -- Peak VRAM: "
        + ", ".join(
            f"{device} {torch.cuda.max_memory_allocated(device) / 1024**3:.3f} GB"
            for device in sorted(used, key=str)
            if device.type == "cuda"
        )
    )
    if not args.dry_run:
        print(f" -- Wrote {len(targets)} tensors to {args.out_dir}")
