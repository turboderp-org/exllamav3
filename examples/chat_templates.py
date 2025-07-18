
class PromptFormat:
    def __init__(self, user_name, bot_name):
        self.user_name = user_name
        self.bot_name = bot_name
    def default_system_prompt(self, think):
        raise NotImplementedError()
    def format(self, system_prompt, messages):
        raise NotImplementedError()
    def add_bos(self):
        raise NotImplementedError()
    def thinktag(self):
        return "<think>", "</think>"


class PromptFormat_raw(PromptFormat):
    description = "Model-agnostic mode simulating a raw chatlog"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
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

    def default_system_prompt(self, think):
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

    def default_system_prompt(self, think):
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
        return False

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|im_end|>"),
            """<|im_end|>"""
        ]


class PromptFormat_phi(PromptFormat):
    description = "Phi3/Phi4 instruct models (note, some Phi4 models use ChatML)"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
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

    def default_system_prompt(self, think):
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

    def default_system_prompt(self, think):
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

    def default_system_prompt(self, think):
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

class PromptFormat_reka(PromptFormat):
    description = "Reka Flash 3"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
        return ""

    def format(self, system_prompt, messages):
        context = ""
        first = True
        for (u, a) in messages:
            if first and system_prompt:
                context += f"human: {system_prompt} {u} <sep> "
                first = False
            else:
                context += f"human: {u} <sep> "
            context += f"assistant:"
            if a is not None: context += f" {a} <sep> "
        return context

    def add_bos(self):
        return False

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            "<sep>",
        ]

    def thinktag(self):
        return " <reasoning>\n", "</reasoning>"


class PromptFormat_cohere(PromptFormat):
    description = "Cohere"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
        return (
            "## Task and Context\n"
            "You help people answer their questions and other requests interactively. You will be asked a very "
            "wide array of requests on all kinds of topics. You should focus on serving the user's needs as "
            "best you can, which will be wide-ranging.\n\n"
            "## Style Guide\n"
            "Unless the user asks for a different style of answer, you should answer in full sentences, using "
            "proper grammar and spelling."
        )

    def format(self, system_prompt, messages):
        context = ""
        if system_prompt:
            context += "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
            context += system_prompt
            context += "<|END_OF_TURN_TOKEN|>"
        for (u, a) in messages:
            context += "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            context += u
            context += "<|END_OF_TURN_TOKEN|>"
            context += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            if a is not None:
                context += a
                context += "<|END_OF_TURN_TOKEN|>"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            "<|END_OF_TURN_TOKEN|>",
        ]


class PromptFormat_dots(PromptFormat):
    description = "Dots"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
        return "You are a helpful assistant."

    def format(self, system_prompt, messages):
        context = ""
        if system_prompt:
            context += "<|system|>"
            context += system_prompt
            context += "<|endofsystem|>"
        for (u, a) in messages:
            context += "<|userprompt|>"
            context += u
            context += "<|endofuserprompt|>"
            context += "<|response|>"
            if a is not None:
                context += a
                context += "<|endofresponse|>"
        return context

    def add_bos(self):
        return False

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|endofresponse|>"),
            "<|endofresponse|>",
        ]

    def thinktag(self):
        return "<|sec-cot|>", "<|sec-end-cot|>"


class PromptFormat_ernie(PromptFormat):
    description = "Ernie"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
        return "You are a helpful assistant."

    def format(self, system_prompt, messages):
        context = "<|begin_of_sentence|>"
        if system_prompt:
            context += system_prompt
            context += "\n"
        for (u, a) in messages:
            context += "User: "
            context += u
            context += "\n"
            context += "Assistant: "
            if a is not None:
                context += a
                context += "<|end_of_sentence|>"
        return context

    def add_bos(self):
        return True

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|end_of_sentence|>")
        ]


class PromptFormat_smollm3(PromptFormat):
    description = "SmolLM3 format (ChatML)"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
        from datetime import datetime
        today_str = datetime.today().strftime("%d %B %Y")
        sp = (
            "## Metadata\n"
            "\n"
            "Knowledge Cutoff Date: June 2025\n"
            f"Today Date: {today_str}\n"
            f"Reasoning Mode: {'/think' if think else '/no_think'}\n"
            "\n"
            "## Custom Instructions\n"
            "\n"
        )
        if think:
            sp += (
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves "
                "thoroughly exploring questions through a systematic thinking process before providing the final precise "
                "and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, "
                "exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking "
                "process. Please structure your response into two main sections: Thought and Solution using the specified "
                "format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning "
                "process in steps. Each step should include detailed considerations such as analysing questions, "
                "summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, "
                "refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, "
                "explorations, and reflections from the Thought section, systematically present the final solution that "
                "you deem correct. The Solution section should be logical, accurate, and concise and detail necessary "
                "steps needed to reach the conclusion.\n\n"
            )
        else:
            sp += "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n"
        return sp

    def format(self, system_prompt, messages):
        context = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for (u, a) in messages:
            context += f"<|im_start|>user\n{u}<|im_end|>\n"
            context += f"<|im_start|>assistant\n"
            if a is not None: context += f"{a}<|im_end|>\n"
        return context

    def add_bos(self):
        return False

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|im_end|>"),
            """<|im_end|>"""
        ]


class PromptFormat_commanda(PromptFormat):
    description = "Command-A (Cohere variant)"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
        # Abridged version
        return (
            "You are a helpful AI assistant.\n"
            "\n"
            "Your information cutoff date is June 2024.\n"
            "\n"
            "You help people answer their questions and other requests interactively. You will be asked a very "
            "wide array of requests on all kinds of topics. Focus on serving the user's needs as best you can, "
            "which will be wide-ranging.\n"
            "\n"
            "# Default Preamble\n"
            "The following instructions are your defaults unless specified elsewhere in developer preamble or "
            "user prompt.\n"
            "- Your name is Command.\n"
            "- You are a large language model built by Cohere.\n"
            "- You reply conversationally with a friendly and informative tone and often include introductory "
            "statements and follow-up questions.\n"
            "- If the input is ambiguous, ask clarifying follow-up questions.\n"
            "- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or "
            "italics, create tables, or format code blocks).\n"
            "- Avoid using LaTeX to generate mathematical notation for complex equations.\n"
            "- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which "
            "case complete the list.\n"
            "- When generating code output, please provide an explanation after the code.\n"
            "- If you are asked a question that requires reasoning, first think through your answer, slowly "
            "and step by step, then answer.\n"
        )

    def format(self, system_prompt, messages):
        context = ""
        if system_prompt:
            context += "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
            context += system_prompt
            context += "<|END_OF_TURN_TOKEN|>"
        for (u, a) in messages:
            context += "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            context += u
            context += "<|END_OF_TURN_TOKEN|>"
            context += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
            if a is not None:
                context += a
                context += "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
        return context

    def add_bos(self):
        return True  # HF template uses double BOS token

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("<|END_RESPONSE|>"),
            tokenizer.single_id("<|END_OF_TURN_TOKEN|>"),
            "<|END_OF_TURN_TOKEN|>",
        ]


class PromptFormat_exaone(PromptFormat):
    description = "EXAONE (EXAONE4)"

    def __init__(self, *args):
        super().__init__(*args)

    def default_system_prompt(self, think):
        # Abridged version
        return (
            "You are a helpful AI assistant."
        )

    def format(self, system_prompt, messages):
        context = ""
        if system_prompt:
            context += "[|system|]\n"
            context += system_prompt
            context += "[|endofturn|]\n"
        for (u, a) in messages:
            context += "[|user|]\n"
            context += u
            context += "[|endofturn|]\n"
            context += "[|assistant|]\n"
            if a is not None:
                context += a
                context += "[|endofturn|]\n"
        return context

    def add_bos(self):
        return False  # HF template uses double BOS token

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            tokenizer.single_id("[|endofturn|]"),
            "[|endofturn|]",
        ]


prompt_formats = {
    "raw": PromptFormat_raw,
    "llama3": PromptFormat_llama3,
    "chatml": PromptFormat_chatml,
    "phi": PromptFormat_phi,
    "mistral": PromptFormat_mistral,
    "gemma": PromptFormat_gemma,
    "glm": PromptFormat_glm,
    "reka": PromptFormat_reka,
    "cohere": PromptFormat_cohere,
    "dots": PromptFormat_dots,
    "ernie": PromptFormat_ernie,
    "smollm3": PromptFormat_smollm3,
    "commanda": PromptFormat_commanda,
    "exaone": PromptFormat_exaone,
}
