from ..model.model import Model

class DraftStackModel(Model):
    def __init__(self, *draft_models: Model):
        super().__init__(None)
        self.draft_models = draft_models
        assert not any(m.caps.get("recurrent_states") for m in self.draft_models), \
            "Speculative decoding with recurrent draft model not supported."
        assert not any(isinstance(m, DraftStackModel) for m in self.draft_models), \
            "Cannot nest draft model stacks."
        self.caps["stack_draft"] = True
