# MLA Support Matrix (vLLM Reference Parity)

This document tracks MLA-family model support against vLLM's implementations and records whether support is:

- `Implemented`: architecture/parser/loading path exists in exllamav3
- `Validated`: local runtime checks completed (config parse, model graph, block forward, or full E2E)

## Support policy

A model is considered "supported" only when all of the following are true:

1. Architecture string is registered and resolves to a matching config/model class.
2. Required parser/preprocessor path (including multimodal) is present.
3. Tensor key mapping and projection layout are handled without custom manual patching per model.
4. Runtime validation passes at least block-forward smoke, and preferably end-to-end generation.

---

## MLA-family models in vLLM and exllamav3 status

| Model architecture (vLLM) | exllamav3 status | Validation status | Notes |
|---|---|---|---|
| `DeepseekForCausalLM` | Implemented | Config/model instantiation path available | Alias to DeepSeek V2 path, matching vLLM behavior |
| `DeepseekV2ForCausalLM` | Implemented | Previously validated with local DeepSeek-V2-Lite flow | Uses `DeepseekV2MLAAttention` |
| `DeepseekV3ForCausalLM` | Implemented | Config/model instantiation path available | Alias to DeepSeek V2 path, matching vLLM behavior |
| `GlmMoeDsaForCausalLM` | Implemented | Config/model instantiation path available | Alias to DeepSeek V2 path, matching vLLM behavior |
| `Qwen3_5ForConditionalGeneration` | Implemented | Quantized EN/KO generation smoke validated | Qwen3.5 linear/full attention mix + MM parser path |
| `Qwen3_5MoeForConditionalGeneration` | Implemented | Quantized EN/KO generation smoke validated | Split GDN projections and split MoE experts mapping handled |
| `DeepseekV32ForCausalLM` | Not implemented | Not validated | Requires vLLM-specific V3.2 indexer/cache path (`DeepseekV32Indexer*`) not present in exllamav3 |
| `Glm4MoeLiteForCausalLM` | Not implemented | Not validated | Requires GLM4-MoE-Lite-specific decoder/load-weight path and additional mapping logic |
| `MiniCPM3ForCausalLM` | Not implemented | Not validated | Separate latent attention implementation and model-specific loader path required |
| `OpenPanguForCausalLM` | Not implemented | Not validated | Separate MLA module + config/weight mapping not yet added |
| `KimiLinear*` MLA family | Not implemented | Not validated | Distinct Kimi MLA path and kernels/mappings required |
| `LongCatFlash*` MLA family | Not implemented | Not validated | Distinct LongCat MLA path and model-specific mappings required |

---

## What was added for Qwen3.5 parity

- New architecture module for:
  - `Qwen3_5ForConditionalGeneration`
  - `Qwen3_5MoeForConditionalGeneration`
- Qwen3.5 parser reuse of Qwen3-VL preprocessing path.
- GatedDeltaNet split-projection support (`in_proj_qkv/in_proj_z/in_proj_a/in_proj_b`).
- MoE split expert tensor loading compatibility for Qwen3.5 tensor layout.

---

## Validation caveat

For very large FP16 checkpoints (e.g., 35B class), full end-to-end generation may be constrained by available VRAM.
In such cases, support is reported as implemented with block-forward validation until quantized checkpoints are available for full runtime validation.

Current status note (2026-02-26): Qwen3.5 35B A3B quantization completed. Split GDN projection handling was corrected and quantized EN/KO generation smoke tests now pass.
