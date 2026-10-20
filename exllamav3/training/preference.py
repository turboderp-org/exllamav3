"""
Preference-optimization losses (DPO / KTO / SimPO / ORPO) for QLoRA-on-EXL3.

Pure-torch loss functions over per-sequence completion log-probabilities, as
produced by ``NativeLlamaQLoRA.compute_logps`` (policy) and -- for the
reference-based objectives -- the same call under
``NativeLlamaQLoRA.adapters_disabled()`` (the frozen-base reference model --
the PEFT disable-adapter trick, so no second model is ever loaded). SimPO and
ORPO are reference-free: they need the policy logps and completion token
counts only.

Semantics follow HuggingFace TRL's stable trainers so hyperparameters and
results transfer directly:

  * DPO -- ``trl.DPOTrainer`` (Rafailov et al. 2023, arXiv:2305.18290), with
    the ``sigmoid`` (+ label smoothing, i.e. cDPO), ``hinge`` (SLiC-HF, Zhao
    et al. 2023) and ``ipo`` (Azar et al. 2023, arXiv:2310.12036) variants.
  * KTO -- ``trl.KTOTrainer`` (Ethayarajh et al. 2024, arXiv:2402.01306),
    promoted to TRL's stable API in huggingface/trl#6175, with the batch-level
    KL estimate from mismatched prompt/completion pairs and the
    ``apo_zero_unpaired`` variant (D'Oosterlinck et al. 2024).
  * SimPO -- ``trl.CPOTrainer`` with ``loss_type="simpo"`` (Meng et al. 2024,
    arXiv:2405.14734): reference-free, length-normalized rewards
    ``beta * mean-per-token logp`` with a target margin ``gamma``
    (TRL's ``simpo_gamma``). TRL's optional ``cpo_alpha`` NLL-on-chosen mix
    (CPO, Xu et al. 2024, arXiv:2401.08417) is applied by the trainer, not
    here.

Credit: the formulations here are adapted from HuggingFace TRL
(https://github.com/huggingface/trl, Copyright The HuggingFace Team,
Apache License 2.0). This is an independent reimplementation against the
exllamav3 native training path, not a copy of TRL code; see
``doc/qlora_handoff.md`` (Session 16) for the design notes.

Convention (matching TRL): a sequence's "logps" is the SUM of per-token
log-probabilities over its completion tokens only (prompt masked); rewards are
``beta * (policy_logps - reference_logps)``. All functions are shape-polymorphic
over a batch dimension and CPU-testable (``tests/test_preference.py``).
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


DPO_LOSS_TYPES = ("sigmoid", "hinge", "ipo")
KTO_LOSS_TYPES = ("kto", "apo_zero_unpaired")
ORPO_NLL_MODES = ("full", "completion")   # what the ORPO SFT term supervises


def dpo_loss(
    policy_chosen_logps: torch.Tensor,       # [b]
    policy_rejected_logps: torch.Tensor,     # [b]
    reference_chosen_logps: torch.Tensor,    # [b]
    reference_rejected_logps: torch.Tensor,  # [b]
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    chosen_counts: Optional[torch.Tensor] = None,    # [b] completion tokens (ipo)
    rejected_counts: Optional[torch.Tensor] = None,  # [b]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss over paired (chosen, rejected) rows.

    Returns ``(losses, chosen_rewards, rejected_rewards)``, each ``[b]``;
    rewards are detached (logging/accuracy only). ``loss_type``:

      * ``"sigmoid"`` (default): ``-log σ(beta·Δ)`` with ``Δ = (π_c - ref_c) -
        (π_r - ref_r)``; ``label_smoothing`` ε > 0 gives the cDPO/robust form
        ``-(1-ε)·log σ(beta·Δ) - ε·log σ(-beta·Δ)``.
      * ``"hinge"`` (SLiC): ``relu(1 - beta·Δ)``.
      * ``"ipo"``: ``(Δ̄ - 1/(2·beta))²`` on length-NORMALIZED logratios
        (``Δ̄`` uses per-token averages; pass ``chosen_counts`` /
        ``rejected_counts`` from ``compute_logps``).
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    if loss_type == "ipo":
        if chosen_counts is None or rejected_counts is None:
            raise ValueError("loss_type='ipo' needs chosen_counts/rejected_counts "
                             "(completion token counts) for length normalization")
        delta_avg = (chosen_logratios / chosen_counts.clamp(min=1)
                     - rejected_logratios / rejected_counts.clamp(min=1))
        losses = (delta_avg - 1.0 / (2.0 * beta)) ** 2
    else:
        delta = chosen_logratios - rejected_logratios
        if loss_type == "sigmoid":
            losses = (-(1.0 - label_smoothing) * F.logsigmoid(beta * delta)
                      - label_smoothing * F.logsigmoid(-beta * delta))
        elif loss_type == "hinge":
            losses = torch.relu(1.0 - beta * delta)
        else:
            raise ValueError(f"unknown DPO loss_type '{loss_type}' "
                             f"(expected one of {DPO_LOSS_TYPES})")

    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()
    return losses, chosen_rewards, rejected_rewards


def simpo_loss(
    policy_chosen_logps: torch.Tensor,    # [b] SUMMED completion logps
    policy_rejected_logps: torch.Tensor,  # [b]
    chosen_counts: torch.Tensor,          # [b] completion token counts
    rejected_counts: torch.Tensor,        # [b]
    beta: float = 2.0,
    gamma: float = 0.5,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple Preference Optimization loss over paired (chosen, rejected) rows
    (SimPO, Meng et al. 2024, arXiv:2405.14734; TRL ``CPOTrainer`` with
    ``loss_type="simpo"`` semantics).

    REFERENCE-FREE: the implicit reward is the length-normalized policy
    logprob ``beta * logps / |y|`` -- no frozen-base forward is involved.
    Inputs are the SUMMED completion logps and token counts exactly as
    returned by ``compute_logps``; normalization happens here.

    The loss is ``-log σ(beta·Δ̄ - gamma)`` with ``Δ̄`` the difference of the
    per-token mean logps and ``gamma`` the target reward margin (TRL's
    ``simpo_gamma``; the paper's γ). ``label_smoothing`` ε > 0 gives the
    cDPO-style robust form (matching TRL's CPO sigmoid smoothing):
    ``-(1-ε)·log σ(beta·Δ̄ - gamma) - ε·log σ(-(beta·Δ̄ - gamma))``.

    Returns ``(losses, chosen_rewards, rejected_rewards)``, each ``[b]``;
    rewards are ``beta * mean-per-token logp``, detached (logging/accuracy
    only). Note the step-0 loss has NO fixed anchor (unlike DPO's ln 2): with
    zero margin it sits at ``-log σ(-gamma)``, and the actual value depends on
    the data through Δ̄.
    """
    chosen_avg = policy_chosen_logps / chosen_counts.clamp(min=1)
    rejected_avg = policy_rejected_logps / rejected_counts.clamp(min=1)
    logits = beta * (chosen_avg - rejected_avg) - gamma
    losses = (-(1.0 - label_smoothing) * F.logsigmoid(logits)
              - label_smoothing * F.logsigmoid(-logits))
    chosen_rewards = (beta * chosen_avg).detach()
    rejected_rewards = (beta * rejected_avg).detach()
    return losses, chosen_rewards, rejected_rewards


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """``log(1 - exp(x))`` for ``x <= 0``, numerically stable (Mächler 2012):
    ``log(-expm1(x))`` near 0, ``log1p(-exp(x))`` far below. ``x`` is
    clamped a hair below 0 so an (im)possible exact-1 probability gives a
    large finite value instead of ``-inf`` and a NaN gradient."""
    x = x.clamp(max=-1e-7)
    return torch.where(x < -0.6931471805599453,
                       torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def orpo_loss(
    policy_chosen_logps: torch.Tensor,    # [b] SUMMED completion logps
    policy_rejected_logps: torch.Tensor,  # [b]
    chosen_counts: torch.Tensor,          # [b] completion token counts
    rejected_counts: torch.Tensor,        # [b]
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Odds-ratio term of ORPO (Hong et al. 2024, arXiv:2403.07691; TRL
    ``ORPOTrainer.odds_ratio_loss`` semantics) over paired (chosen, rejected)
    rows. REFERENCE-FREE.

    With ``P(y|x) = exp(mean per-token logp)`` (the paper's length-normalized
    probability, eq. 3) and ``odds(y|x) = P / (1 - P)``, the loss is

        L_OR = -beta * log σ( log odds(y_w|x) - log odds(y_l|x) )

    where ``log odds = logp_avg - log(1 - exp(logp_avg))``. ``beta`` is the
    paper's lambda (0.1 for Mistral, 0.2 Llama-2, 0.25 Phi-2; TRL default
    0.1). The full ORPO objective is ``NLL(chosen) + L_OR``; the trainer adds
    the NLL (see ``ORPO_NLL_MODES``).

    Inputs are the SUMMED completion logps and token counts exactly as
    returned by ``compute_logps``; normalization happens here. Returns
    ``(losses, chosen_rewards, rejected_rewards, log_odds)``, each ``[b]``:
    rewards are ``beta * mean-per-token logp`` (TRL's logging convention) and
    ``log_odds`` the log odds ratio, all detached. Gradient note (paper
    sec. 4.3): dL_OR scales with ``1 / (1 + odds_w / odds_l)``, so the term
    goes quiet once chosen is already much more likely than rejected.
    """
    chosen_avg = policy_chosen_logps / chosen_counts.clamp(min=1)
    rejected_avg = policy_rejected_logps / rejected_counts.clamp(min=1)
    log_odds = ((chosen_avg - rejected_avg)
                - (log1mexp(chosen_avg) - log1mexp(rejected_avg)))
    losses = -beta * F.logsigmoid(log_odds)
    chosen_rewards = (beta * chosen_avg).detach()
    rejected_rewards = (beta * rejected_avg).detach()
    return losses, chosen_rewards, rejected_rewards, log_odds.detach()


def kto_loss(
    policy_chosen_logps: torch.Tensor,        # [n_d]  (desirable rows)
    policy_rejected_logps: torch.Tensor,      # [n_u]  (undesirable rows)
    policy_kl_logps: Optional[torch.Tensor],  # [n_kl] mismatched pairs (or None)
    reference_chosen_logps: torch.Tensor,     # [n_d]
    reference_rejected_logps: torch.Tensor,   # [n_u]
    reference_kl_logps: Optional[torch.Tensor],  # [n_kl]
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    loss_type: str = "kto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kahneman-Tversky Optimization loss over UNPAIRED rows labeled
    desirable/undesirable (KTO paper eq. 7; TRL ``KTOTrainer`` semantics).

    Either subset may be empty for a given micro-batch (0-length tensors are
    fine). The reference point ``kl`` is the batch estimate of
    ``KL(π||ref)`` from MISMATCHED prompt/completion pairs (see
    :func:`mismatched_kl_shift`): ``mean(policy_kl_logps - reference_kl_logps)``,
    detached and clamped to ``>= 0``. Single-process estimate -- under DDP,
    all-reduce it across ranks before use (not wired here).

      * ``"kto"``:  chosen ``1 - σ(beta·(logratio - kl))``,
                    rejected ``1 - σ(beta·(kl - logratio))``.
      * ``"apo_zero_unpaired"``: no KL term; chosen ``1 - σ(beta·logratio)``,
                    rejected ``σ(beta·logratio)``.

    Returns ``(losses, chosen_rewards, rejected_rewards, kl)`` where ``losses``
    is the concatenation ``[desirable_weight·chosen ; undesirable_weight·
    rejected]`` (reduce with ``.mean()``/``.nanmean()``); rewards and ``kl``
    are detached.
    """
    if loss_type not in KTO_LOSS_TYPES:
        raise ValueError(f"unknown KTO loss_type '{loss_type}' "
                         f"(expected one of {KTO_LOSS_TYPES})")

    any_t = (policy_chosen_logps if policy_chosen_logps.numel()
             else policy_rejected_logps)
    if (loss_type == "kto" and policy_kl_logps is not None
            and policy_kl_logps.numel()):
        kl = (policy_kl_logps - reference_kl_logps).mean().detach().clamp(min=0)
        kl = kl.to(any_t.dtype)
    else:
        kl = any_t.new_zeros(())

    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    if loss_type == "kto":
        chosen_losses = 1.0 - torch.sigmoid(beta * (chosen_logratios - kl))
        rejected_losses = 1.0 - torch.sigmoid(beta * (kl - rejected_logratios))
    else:  # apo_zero_unpaired
        chosen_losses = 1.0 - torch.sigmoid(beta * chosen_logratios)
        rejected_losses = torch.sigmoid(beta * rejected_logratios)

    losses = torch.cat((desirable_weight * chosen_losses,
                        undesirable_weight * rejected_losses), dim=0)
    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()
    return losses, chosen_rewards, rejected_rewards, kl


def mismatched_kl_shift(n: int) -> list[int]:
    """Index permutation pairing each prompt with ANOTHER example's completion
    for the KTO KL estimate: TRL's "+1 offset" rotation
    (``[n-1, 0, 1, ..., n-2]``, i.e. prompt i gets completion i-1). With
    ``n == 1`` the pair is matched, which would bias the KL estimate toward the
    policy's own reward -- callers should skip the KL term for singleton
    batches (the trainer warns; TRL requires batch size > 1 for the same
    reason)."""
    if n <= 0:
        return []
    return [n - 1] + list(range(n - 1))
