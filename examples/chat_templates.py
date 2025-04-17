
class PromptFormat:
    def __init__(self, user_name, bot_name):
        self.user_name = user_name
        self.bot_name = bot_name
    def default_system_prompt(self):
        raise NotImplementedError()
    def format(self, system_prompt, messages):
        raise NotImplementedError()
    def add_bos(self):
        raise NotImplementedError()


class PromptFormat_raw(PromptFormat):
    description = "Model-agnostic mode simulating a raw chatlog"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self):
        return (
            f"This is a conversation between a helpful AI assistant " +
            (f"named {self.bot_name} " if self.bot_name != "Assistant" else "") +
            (f"and a user named {self.user_name}." if self.user_name != "User" else """and a user.""")
        )

    def format(self, system_prompt, messages):
        context = system_prompt + "\n"
        for (u, a) in messages:
            context += f"{self.user_name}: {u}\n"
            context += f"{self.bot_name}:"
            if a is not None:
                context += f"{a}\n"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            self.user_name + ":",
            self.user_name[0:1] + ":",
            self.user_name.upper() + ":",
            self.user_name.lower() + ":",
            tokenizer.eos_token_id
        ]


class PromptFormat_llama3(PromptFormat):
    description = "Llama3-instruct models"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self):
        return (
            """Assist users with tasks and answer questions to the best of your knowledge. Provide helpful and informative """
            """responses. Be conversational and engaging. If you are unsure or lack knowledge on a topic, admit it and try """
            """to find the answer or suggest where to find it. Keep responses concise and relevant. Follow ethical """
            """guidelines and promote a safe and respectful interaction."""
        )

    def format(self, system_prompt, messages):
        context = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        for (u, a) in messages:
            context += f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
            context += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            if a is not None: context += f"{a}<|eot_id|>"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|eot_id|>"),
            tokenizer.single_id("<|start_header_id|>")
        ]


class PromptFormat_chatml(PromptFormat):
    description = "ChatML format, as used by e.g. Qwen"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self):
        return (
            f"You are {self.bot_name}, a large language model. Answer as concisely as possible."
        )

    def format(self, system_prompt, messages):
        context = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for (u, a) in messages:
            context += f"<|im_start|>user\n{u}<|im_end|>\n"
            context += f"<|im_start|>assistant\n"
            if a is not None: context += f"{a}<|im_end|>\n"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|im_end|>"),
            """<|im_end|>"""
        ]


class PromptFormat_phi(PromptFormat):
    description = "Phi3/Phi4 instruct models"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self):
        return (
            f"You are a helpful AI assistant."
        )

    def format(self, system_prompt, messages):
        context = f"<|system|>\n{system_prompt}<|end|>\n"
        for (u, a) in messages:
            context += f"<|user|>\n{u}<|end|>\n"
            context += f"<|assistant|>\n"
            if a is not None: context += f"{a}<|end|>\n"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|end|>"),
        ]


class PromptFormat_glm(PromptFormat):
    description = "ChatGLM(4) models"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self):
        return (
            f"You are a helpful AI assistant."
        )

    def format(self, system_prompt, messages):
        context = f"[gMASK]<sop><|system|>\n{system_prompt}"
        for (u, a) in messages:
            context += f"<|user|>\n{u}"
            context += f"<|assistant|>\n"
            if a is not None: context += f"{a}"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|user|>"),
        ]


class PromptFormat_mistral(PromptFormat):
    description = "Mistral-instruct models (v3)"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self):
        return (
            """You are a helpful AI assistant."""
        )

    def format(self, system_prompt, messages):
        context = ""
        first = True
        for (u, a) in messages:
            if first:
                context += f"[INST] {system_prompt}\n\n{u}[/INST]"
                first = False
            else:
                context += f"[INST] {u}[/INST]"
            if a is not None: context += f" {a}</s>"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id
        ]


class PromptFormat_gemma(PromptFormat):
    description = "Gemma"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self):
        return ""

    def format(self, system_prompt, messages):
        context = ""
        for (u, a) in messages:
            context += f"<start_of_turn>user\n"
            context += f"{u}<end_of_turn>\n"
            context += f"<start_of_turn>model\n"
            if a is not None: context += f"{a}<end_of_turn>\n"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<end_of_turn>"),
            tokenizer.single_id("<start_of_turn>"),
        ]


prompt_formats = {
    "raw": PromptFormat_raw,
    "llama3": PromptFormat_llama3,
    "chatml": PromptFormat_chatml,
    "phi": PromptFormat_phi,
    "mistral": PromptFormat_mistral,
    "gemma": PromptFormat_gemma,
    "glm": PromptFormat_glm,
}
