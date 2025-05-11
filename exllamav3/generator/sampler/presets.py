from .custom import *

class DefaultSampler(CustomSampler):
    """
    Sensible default for most models
    """
    def __init__(self):
        super().__init__([
            SS_MinP(0.08),
            SS_Temperature(0.8),
            SS_Sample()
        ])

class ArgmaxSampler(CustomSampler):
    """
    Returns top token
    """
    def __init__(self):
        super().__init__([
            SS_Argmax()
        ])

GreedySampler = ArgmaxSampler

class CategoricalSampler(CustomSampler):
    """
    Samples from unmodified categorical distribution
    """
    def __init__(self, temperature: float = 1.0):
        if temperature == 0:
            super().__init__([
                SS_Argmax()
            ])
        else:
            super().__init__([
                SS_Temperature(temperature),
                SS_Sample()
            ])

GumbelSampler = CategoricalSampler

class TopKSampler(CustomSampler):
    """
    Truncates distribution to top_k values before sampling
    """
    def __init__(self, top_k: int, temperature: float = 1.0):
        assert top_k >= 1
        if top_k == 1 or temperature == 0:
            super().__init__([
                SS_Argmax()
            ])
        else:
            super().__init__([
                SS_Temperature(temperature),
                SS_TopK(top_k),
                SS_Sample()
            ])

class TopPSampler(CustomSampler):
    """
    Truncates distribution to the top probabilities <= top_p (at least 1 candidate) before sampling
    """
    def __init__(self, top_p: float, temperature: float = 1.0, temperature_last = False):
        if top_p == 0 or temperature == 0:
            super().__init__([
                SS_Argmax()
            ])
        else:
            if temperature_last:
                super().__init__([
                    SS_TopP(top_p),
                    SS_Temperature(temperature),
                    SS_Sample()
                ])
            else:
                super().__init__([
                    SS_Temperature(temperature),
                    SS_TopP(top_p),
                    SS_Sample()
                ])

class ComboSampler(CustomSampler):
    """
    Single class with an argument for each sampling step
    """
    def __init__(
        self,
        rep_p: float = 1.0,
        freq_p: float = 0.0,
        pres_p: float = 0.0,
        rep_sustain_range: int = int(10e7),
        rep_decay_range: int = 0,
        temperature: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        temp_last: bool = False,
    ):
        # Steps with default parameters become no-ops
        stack = [
            SS_RepP(rep_p, rep_sustain_range, rep_decay_range),
            SS_PresFreqP(pres_p, freq_p, rep_sustain_range, rep_decay_range),
        ]

        if temperature == 0.0 or top_k == 1:
            stack += [
                SS_Argmax()
            ]
        else:
            stack += [
                SS_Temperature(temperature if not temp_last else 1.0),
                SS_MinP(min_p),
                SS_TopK(top_k),
                SS_TopP(top_p),
                SS_Temperature(temperature if temp_last else 1.0),
                SS_Sample()
            ]

        super().__init__(stack)