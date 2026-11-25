"""Paired shared/latent projections for single-token Olala EXL3 inference."""

import torch
import triton
import triton.language as tl

from ...ext import exllamav3_ext as ext


@triton.jit
def _project_inputs(X, Weight, Latent, Shared, Hidden: tl.constexpr, LatentSize: tl.constexpr,
                    Rows: tl.constexpr):
    rows = tl.program_id(0) * Rows + tl.arange(0, Rows)
    k = tl.arange(0, triton.next_power_of_2(Hidden))
    x = tl.load(X + k, k < Hidden, 0).to(tl.float32)
    w = tl.load(Weight + rows[:, None] * Hidden + k[None, :], k[None, :] < Hidden, 0).to(tl.float32)
    projected = tl.sum(w * x[None, :], 1)
    if tl.program_id(0) < LatentSize // Rows:
        tl.store(Latent + rows, projected)
    else:
        # Olala's shared up projection produces FP16 before ReLU squared.
        positive = tl.maximum(projected.to(tl.float16).to(tl.float32), 0.)
        tl.store(Shared + rows - LatentSize, positive * positive)


@triton.jit
def _project_outputs(Routed, Shared, Weight, Scale, Output, LatentSize: tl.constexpr,
                     SharedSize: tl.constexpr, Rows: tl.constexpr):
    rows = tl.program_id(0) * Rows + tl.arange(0, Rows)
    weights = Weight + rows[:, None] * (LatentSize + SharedSize)
    k = tl.arange(0, triton.next_power_of_2(LatentSize))
    h = tl.arange(0, triton.next_power_of_2(SharedSize))
    x = tl.load(Routed + k, k < LatentSize, 0).to(tl.float16).to(tl.float32)
    u = tl.load(Shared + h, h < SharedSize, 0).to(tl.float32)
    a = tl.load(weights + k[None, :], k[None, :] < LatentSize, 0).to(tl.float32)
    b = tl.load(weights + LatentSize + h[None, :], h[None, :] < SharedSize, 0).to(tl.float32)
    routed = tl.sum(a * x[None, :], 1) * tl.load(Scale)
    tl.store(Output + rows, routed + tl.sum(b * u[None, :], 1))


def _weight(linear):
    inner = linear.inner
    weight = inner.trellis.new_empty((inner.in_features, inner.out_features), dtype=torch.half)
    ext.reconstruct_had_slice(weight, inner.trellis, inner.suh, inner.svh,
                              inner.K, inner.mcg, inner.mul1, 0)
    return weight.T


def pack_projections(linears):
    up = torch.cat([_weight(linear) for linear in linears[:2]], dim = 0).contiguous()
    down = torch.cat([_weight(linear) for linear in linears[2:]], dim = 1).contiguous()
    return up, down


def decode_moe(moe, x, params):
    hidden = x.shape[-1]
    latent = moe.config.latent_size
    width = moe.config.shared_size
    up, down = moe.decode_weights
    rows = 4
    x = x.view(1, hidden)
    indices, routing_weights = moe.routing_fn(1, moe.routing_cfg, x, params)
    latent_input = x.new_empty((1, latent), dtype = torch.float32)
    shared_activation = x.new_empty((1, width))
    _project_inputs[((latent + width) // rows,)](
        x, up, latent_input, shared_activation, hidden, latent, rows, enable_fp_fusion = False,
    )
    expert_input, scale = moe.latent_in.scale_input(latent_input, params)
    moe.bc.run_bszN(expert_input, indices, routing_weights)
    result = x.new_empty((1, 1, hidden), dtype = torch.float32)
    _project_outputs[(hidden // rows,)](
        moe.experts_cfg.out_bszn, shared_activation, down, scale, result,
        latent, width, rows, enable_fp_fusion = False,
    )
    return result
