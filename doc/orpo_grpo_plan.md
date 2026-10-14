# ORPO and GRPO for the EXL3 QLoRA trainer (plan, 2026-10-14)

Status: **ORPO built and CPU-tested** (branch `feat/orpo`, handoff Session 60; box
list there). GRPO not started.

Requested for the user's friend Neko. Queue order: ORPO first, then GRPO.
Both train the LoRA adapter on the frozen EXL3 base; neither loads a second
model. This doc holds the research (paper, TRL 1.15.0, axolotl 0.20.0) the
implementations follow, so the semantics can be checked against it.

## How the references handle "the other models"

| Trainer | Reference model | Old policy | Notes |
|---|---|---|---|
| TRL `DPOTrainer` | PEFT: adapters disabled on the same model. Otherwise a second full copy. | n/a | Same trick our `preference.py` already uses. |
| TRL `ORPOTrainer` (now `trl.experimental.orpo`) | none, reference-free by construction | n/a | One forward over chosen+rejected concatenated. |
| TRL `GRPOTrainer` | `beta == 0` (the DEFAULT): none at all. PEFT: adapters disabled (or a saved "ref" adapter when retraining an existing one). Else a second full copy loaded from the model path. | `num_iterations == 1` (default): `per_token_logps.detach()`, no extra forward. `> 1`: one no-grad forward on the training model after rollout. With vLLM: always recomputed on the training model, plus importance-sampling correction against vLLM's own sampling logps (`vllm_importance_sampling_correction=True`, mode `sequence_mask`). | Generation via HF `generate` or vLLM (server / colocate) with weight sync. |
| axolotl `rl: orpo` | its own in-trainer loss (`AxolotlTrainer.orpo_compute_loss`, `orpo_alpha`) or TRL's experimental ORPOTrainer | n/a | Same math as TRL. |
| axolotl `rl: grpo` | wraps TRL `GRPOTrainer` (+ async / fast-async subclasses), so identical | identical | `trl.reward_funcs: ["module.func"]` loaded with `importlib`; `rollout_func` likewise. |
| DeepSeekMath paper (GRPO, arXiv:2402.03300) | pi_ref is the initial SFT model, RESET to the current policy at every outer iteration (Algorithm 1) | snapshot used to sample | G=64, beta=0.04, lr 1e-6, batch 1024, max 1024 tokens; mu effectively 1. |
| ORPO paper (arXiv:2403.07691) | none | n/a | lambda 0.1 (Mistral), 0.2 (Llama-2), 0.25 (Phi-2); ablated {0.1, 0.5, 1.0}. Starts from PRETRAINED bases, replacing the SFT stage. |

So the two-or-three-model picture is the full-finetune case. For adapters
every framework collapses to one loaded model, exactly our
`adapters_disabled()` trick; GRPO's "old policy" is a detached tensor.

## ORPO semantics (TRL `trl/experimental/orpo/orpo_trainer.py`)

* Paired data, same `prompt / chosen / rejected` format as DPO and SimPO.
* `logps` are the MEAN per-token completion log-probs (`average_log_prob=True`,
  prompt labels `-100`), i.e. our `compute_logps` sum / counts.
* `log_odds = (lc - lr) - (log1mexp(lc) - log1mexp(lr))` with
  `log1mexp(a) = log(1 - exp(a))`, a <= 0; use `log1p(-exp(a))` with the
  usual `a > -0.693` branch (`log(-expm1(a))`) for accuracy.
* `L_OR = -beta * logsigmoid(log_odds)`, `beta` = the paper's lambda, TRL
  default 0.1 (axolotl `orpo_alpha`).
* NLL term: TRL and axolotl both compute the chosen NLL over the FULL
  sequence, prompt included (`labels = input_ids where attention_mask == 1`,
  comment "orpo chosen nll loss is computed over the full prompt and
  response"), as a batch-token mean. The paper calls it the standard causal
  NLL over the "reference tokens" and does not spell out prompt masking.
  Plan: `orpo_nll: full | completion` with `full` the default to match
  TRL/axolotl numerically; `completion` for people who want the SFT term to
  supervise only the answer.
* Loss = `nll - beta * mean(logsigmoid(log_odds))`. Rewards for logging =
  `beta * mean logp`. Extra metrics: `log_odds_ratio` (mean logsigmoid),
  `log_odds_chosen` (mean log_odds), nll.
* Gradient note (paper): dL_OR scales with `1 / (1 + odds_w/odds_l)`, so the
  term goes quiet once the model already prefers chosen.
* Where it lands: `exllamav3/training/preference.py` (`orpo_loss`),
  `training/qlora_train_pref.py` (`--method orpo`, `orpo_batch_metrics`;
  `compute_logps` needs an option to also return the full-sequence NLL,
  or a second masked call), the YAML config, `realtime.py` ingest
  (`{"prompt","chosen","rejected"}` already routes to DPO; add
  `pref_method` selection), `tests/test_preference.py`, README table.

## GRPO semantics (TRL `grpo_trainer.py`, `grpo_config.py` defaults)

Defaults (TRL 1.15.0): `beta 0.0`, `epsilon 0.2`, `epsilon_high None` (=
epsilon), `num_iterations 1`, `loss_type "dapo"`, `scale_rewards "group"`,
`num_generations 8`, `temperature 1.0`, `top_p 1.0`, `top_k 0`,
`max_completion_length 512`, `importance_sampling_level "token"`,
`mask_truncated_completions False`.

Per rollout batch:
1. Sample `G` completions per prompt from the current policy.
2. Rewards: every reward function is called as
   `f(prompts=..., completions=..., completion_ids=..., **extra_columns)`
   and returns `list[float | None]` (None -> NaN, excluded); weighted sum
   over functions (`reward_weights`).
3. Advantages: `A_i = r_i - mean_group(r)`, divided by `std_group(r) + 1e-4`
   when `scale_rewards == "group"` (`"batch"`: batch std; `"none"`: no
   scaling, Dr. GRPO style). NaN -> 0.
4. Per-token logps on the training forward for the policy; `old` is
   `detach()` of the same unless `num_iterations > 1`; reference logps only
   when `beta != 0`, via the adapters-disabled forward.
5. Loss per token: `ratio = exp(logp - old)`, clipped
   `[1 - eps_low, 1 + eps_high]`, `-min(ratio*A, clipped*A)`; plus
   `beta * k3` with `k3 = exp(ref - logp) - (ref - logp) - 1`.
6. Reduction: `grpo` = per-sequence token mean then batch mean (the paper);
   `dapo` (default) = token sum over the whole generation batch /
   total completion tokens in the batch; `dr_grpo` = sum / (B * max_len);
   `bnpo` = masked token mean over the micro-batch.

Our mapping:
* Rollouts through the native generator after `apply_to_native()` (the
  realtime module's sync), sampling with temperature / top_p / top_k;
  tokens only, never the generator's logprobs (recompute on the training
  forward, as TRL does under vLLM).
* `compute_logps` grows a per-token variant (the fused CE already produces
  the per-token NLL; return it unsummed).
* Reward plugin: `--reward path/to/file.py:func` (several allowed,
  `--reward-weights`), same call signature as TRL so community reward
  functions drop in; built-ins for format / length / exact-match.
* New script `training/qlora_train_grpo.py` + YAML, `exllamav3/training/
  grpo.py` for advantages and the loss (CPU-testable), README, handoff.
* Box proof: a small dense model with a verifiable reward (the reward
  rises; a `beta > 0` run keeps the k3 KL bounded).
