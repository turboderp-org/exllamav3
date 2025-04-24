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
