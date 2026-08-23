from __future__ import annotations
from .dflash import DFlashConfig, DFlashModel

# Meta's "assistant" release for Muse Glimmer is a DFlash drafter under a different name: same
# 5-layer llama-style block-diffusion drafter, tap encoder over target hidden states (outputs of
# target layers target_layer_ids, i.e. the same hidden_states[i + 1] convention as the original
# DFlash release, handled by the inherited +1 shift), mask-token noise window, and the target's
# embeddings and head. The noise block is embedded from the raw lookup table, bypassing the
# target's embedding norm.


class MuseGlimmerAssistantConfig(DFlashConfig):
    arch_string = "MuseGlimmerAssistantModel"

    # The unified HF candidate generator taps hidden_states[i + 1] = output of layer i, which is
    # exllamav3 export index i: raw ids. Verified empirically: raw taps accept 3.00 tok/round vs
    # 2.79 for the +1 variant
    tap_shift = 0

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes = {"text": MuseGlimmerAssistantModel},
            **kwargs
        )

        # HF mask keeps sliding_window including the query
        self.sliding_window = self.sliding_window - 1


class MuseGlimmerAssistantModel(DFlashModel):
    config_class = MuseGlimmerAssistantConfig

    key_fc = "encoder.fc"
    key_fc_norm = "encoder.output_norm_enc"
