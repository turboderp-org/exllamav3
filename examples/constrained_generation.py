import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, Filter, FormatronFilter
from formatron.schemas.pydantic import ClassSchema
from formatron.formatter import FormatterBuilder
from pydantic import conlist
from typing import Literal, Optional
from formatron.extractor import NonterminalExtractor
import json

def get_superhero_filter(tokenizer) -> list[Filter]:

    class SuperheroAppearance(ClassSchema):
        title: str
        issue_number: int
        year: int
    class Superhero(ClassSchema):
        name: str
        secret_identity: str
        gender: Literal["male", "female"]
        superpowers: conlist(str, max_length = 5)
        first_appearance: SuperheroAppearance

    # Create JSON formatter and ExLlama filter
    f = FormatterBuilder()
    f.append_line(f"{f.json(Superhero, capture_name = 'json')}")
    filters = [FormatronFilter(tokenizer, eos_after_completed = True, formatter_builder = f)]

    # Additional constraint to force leading {
    f = FormatterBuilder()
    f.append_line("{")
    filters += [FormatronFilter(tokenizer, formatter_builder = f)]

    # Test triggered filter, triggers on "Bruce" (ID 79579 in Llama3.1 vocab)
    f = FormatterBuilder()
    f.append_str(" Thomas")
    filters += [FormatronFilter(tokenizer, trigger_token = 79579, formatter_builder = f)]

    return filters


def get_arithmetic_filter(tokenizer) -> list[Filter]:

    class ArithmeticExpressionExtractor(NonterminalExtractor):
        def __init__(self, nonterminal: str, capture_name: Optional[str] = None):
            super().__init__(nonterminal, capture_name)

        def extract(self, input_str: str) -> Optional[tuple[str, any]]:
            i = 0
            left_bracket = 0
            while i < len(input_str):
                if input_str[i].isdigit() or input_str[i] in "+-*/.":
                    i += 1
                    continue
                if input_str[i] == "(":
                    i += 1
                    left_bracket += 1
                    continue
                if input_str[i] == ")":
                    i += 1
                    left_bracket -= 1
                    continue
                else:
                    break
            if left_bracket != 0:
                return None
            return input_str[i:], input_str[:i]

        @property
        def kbnf_definition(self) -> str:
            return (
                """expression ::=  term { ("+" | "-") term };"""
                """term       ::= factor { ("*" | "/") factor };"""
                """factor     ::= number | "(" expression ")";"""
                """number     ::= #"[0-9]+(\\\\.[0-9]+)?([eE][+-]?[0-9]+)?";"""
            ).replace("expression", self.nonterminal)

    # Create arithmetic formatter and ExLlama filter
    f = FormatterBuilder()
    extractor1 = f.extractor(lambda nonterminal: ArithmeticExpressionExtractor(nonterminal, 'ex1'))
    extractor2 = f.extractor(lambda nonterminal: ArithmeticExpressionExtractor(nonterminal, 'ex2'))
    f.append_line(f"{extractor1} = {extractor2}")
    filters = [FormatronFilter(tokenizer, eos_after_completed = True, formatter_builder = f)]

    return filters


def stream_gen(generator, tokenizer, prompt, filters):

    # Create job
    job = Job(
        input_ids = tokenizer.encode(prompt, add_bos = True),
        filters = filters,
        max_new_tokens = 400,
    )
    generator.enqueue(job)

    print("----------------------")
    print(prompt, end = "")

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for result in results:
            text = result.get("text", "")
            print(text, end = "", flush = True)
    print()


def main():

    # Load model etc.
    model_dir = "/mnt/str/models/llama3.1-8b-instruct/exl3/4.0bpw/"
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 8192)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer)

    # Single gens can reuse filters
    sh_filter = get_superhero_filter(tokenizer)
    stream_gen(generator, tokenizer, "Here is vital information about Superman, in JSON format:\n\n", sh_filter)
    stream_gen(generator, tokenizer, "Here is vital information about Batman, in JSON format:\n\n", sh_filter)
    ar_filter = get_arithmetic_filter(tokenizer)
    stream_gen(generator, tokenizer, "Number of seconds in a century: 100*", ar_filter)  # (Llama3.1 can't math)
    stream_gen(generator, tokenizer, "Two plus two: 2+", ar_filter)

    # But filters are stateful, so we need multiple instances for batched gen
    sh_filter2 = get_superhero_filter(tokenizer)
    ar_filter2 = get_arithmetic_filter(tokenizer)
    batched_gens = generator.generate(
        prompt = [
            "Here is vital information about Superman, in JSON format:\n\n",
            "Here is vital information about Batman, in JSON format:\n\n",
            "Number of seconds in a century: 100*",
            "Two plus two: 2+",
        ],
        filters = [
            sh_filter,
            sh_filter2,
            ar_filter,
            ar_filter2,
        ],
        max_new_tokens = 400,
        add_bos = True,
    )
    for g in batched_gens:
        print("----------------------")
        print(g)


if __name__ == "__main__":
    main()
