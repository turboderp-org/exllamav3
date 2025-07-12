from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import model_init
from exllamav3 import model_init, Generator, Job
import argparse, contextlib
import torch
import util
import random, math
from datasets import load_dataset
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from exllamav3.util.progress import ProgressBar

@disk_lru_cache("get_dataset_mmlu")
def get_dataset_mmlu(split):
    print(f" -- Loading dataset, split {split}")
    dataset = load_dataset("cais/mmlu", "all", split = split)
    rows = [example for example in dataset]
    return rows

def main(args):

    # Initialize
    model, config, cache, tokenizer = model_init.init(args)
    generator = Generator(
        model = model,
        cache = cache,
        max_batch_size = 1024,
        tokenizer = tokenizer,
        max_q_size = 1,
        show_visualizer = args.visualize_cache
    )

    # Sampling
    c_options = "ABCD"
    token_map = [tokenizer.single_id(piece) for piece in [" " + c for c in c_options]]

    # Get dataset
    dataset_dev = get_dataset_mmlu("dev")
    dataset_all = get_dataset_mmlu("test")
    dataset_dev = sorted(dataset_dev, key = lambda q: q["subject"])
    dataset_all = sorted(dataset_all, key = lambda q: q["subject"])

    all_subjects = set([q["subject"] for q in dataset_dev])
    if args.subjects != "all":
        sel_subjects = args.subjects.split(",")
        for s in sel_subjects:
            if s not in all_subjects:
                print(f" ## Subject {s} is not present in dataset")
                sys.exit()
        all_subjects = set(sel_subjects)

    # Skip
    all_subjects = sorted(list(all_subjects))
    all_subjects = all_subjects[args.skip_subjects:]
    all_subjects = set(all_subjects)

    # Optionally shuffle
    if args.shuffle:
        for problem in dataset_all:
            if problem["subject"] in all_subjects:
                perm = random.sample(range(4), k = 4)
                problem["choices"] = [problem["choices"][i] for i in perm]
                problem["answer"] = perm.index(problem["answer"])

    # Random sample subset of questions
    n_sample = args.random_sample
    if n_sample:
        full_len = len(dataset_all)
        assert n_sample <= full_len, f"Dataset only has {full_len} questions."
        random.seed(0)
        random.shuffle(dataset_all)
        dataset_all = dataset_all[:n_sample]

    # Format
    def format_question(question: str, choices: list[str], answer: int | None):
        f = question + "\n"
        for i, c in enumerate(c_options):
            f += c + ". " + choices[i] + "\n"
        f += "Answer:"
        if answer is not None:
            f += " " + c_options[answer] + "\n\n"
        return f

    # Fewshot preprompts
    preprompt_ids = {}
    with ProgressBar("Preprompts", len(all_subjects), transient = False) as progress:
        for idx, subject in enumerate(all_subjects):
            preprompt = \
                f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
            fewshots = 0
            for pq in dataset_dev:
                if fewshots == args.fewshot_examples: break
                if pq["subject"] != subject: continue
                preprompt += format_question(pq["question"], pq["choices"], pq["answer"])
            preprompt_ids[subject] = tokenizer.encode(preprompt, add_bos = True)
            progress.update(idx + 1)

    # Questions
    total_jobs = 0
    for q in dataset_all:
        if q["subject"] in all_subjects:
            total_jobs += 1

    num_remaining = {s: 0 for s in all_subjects}

    with ProgressBar("Questions", total_jobs, transient = False) as progress:
        for q in dataset_all:
            sub = q["subject"]
            if sub not in all_subjects:
                continue
            if args.max_q_per_subject and num_remaining[sub] >= args.max_q_per_subject:
                continue
            prompt = format_question(q["question"], q["choices"], None)
            prompt_ids = tokenizer.encode(prompt, add_bos = False)
            job = Job(
                input_ids = torch.cat([preprompt_ids[sub], prompt_ids], dim = -1),
                max_new_tokens = 1,
                return_logits = True,
                identifier = q,
            )
            generator.enqueue(job)
            num_remaining[sub] += 1
            progress.update(generator.num_remaining_jobs())

    # Evaluate
    def print_results(p_subject):
        nonlocal dataset_all, n_sample
        total = 0
        correct = 0
        confidence_sum = 0.0
        for q in dataset_all:
            if not "answer_correct" in q:
                continue
            if p_subject is not None and q["subject"] != p_subject:
                continue
            total += 1
            if q["answer_correct"]:
                correct += 1
            confidence_sum += q["correct_answer_confidence"]
        if p_subject is None:
            p_subject = "all subjects"
        score = correct / total
        conf = confidence_sum / total
        if n_sample:
            p_subject = f"all subjects, {total} random samples"
            interval = math.sqrt(score * (1 - score) / total * (full_len - total) / (full_len - 1))
            print(
                f"{p_subject:40}: {correct: 5}/{total: 5} = {score * 100:6.2f}% +/- {interval * 100: 6.2f}%"
                + (" (95% CI)" if total == n_sample else "")
            )
        else:
            print(f"{p_subject:40}: {correct: 5}/{total: 5} = {score * 100:6.2f}% correct, ({conf * 100:6.2f}% prob.)")

    with ProgressBar("Testing", total_jobs, transient = False) as progress:
        last_update = 0
        update_interval = (n_sample or 0) // 10
        while generator.num_remaining_jobs():
            results = generator.iterate()
            tested = total_jobs - generator.num_remaining_jobs()
            for result in results:
                if not result["eos"]:
                    continue
                q = result["identifier"]
                logits = result["logits"][0, 0].float().cpu()
                logits = logits[token_map]
                favored_anwser = torch.argmax(logits, dim = -1).item()
                model_probs = torch.softmax(logits, dim = -1).tolist()
                correct_answer = q["answer"]
                confidence = model_probs[correct_answer]
                q["correct_answer_confidence"] = confidence
                q["answer_correct"] = favored_anwser == correct_answer
                sub = q["subject"]
                num_remaining[sub] -= 1
                if num_remaining[sub] == 0 and not n_sample:
                    print_results(sub)
                progress.update(tested)
            if n_sample and tested > last_update + update_interval:
                last_update = tested
                print_results(None)

    print_results(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run MMLU evaluation")
    model_init.add_args(parser)
    parser.add_argument("-fs", "--fewshot_examples", type = int, default = 5, help = "Number of examples for fewshot examples, max 5")
    parser.add_argument("-sub", "--subjects", type = str, default = "all", help = "Comma-separated list of categories to test, or 'all'")
    parser.add_argument("-shf", "--shuffle", action = "store_true", help = "Shuffle choices randomly")
    parser.add_argument("-vis", "--visualize_cache", action = "store_true", help = "Show cache visualizer (slow)")
    parser.add_argument("-skip", "--skip_subjects", type = int, default = 0, help = "Skip number of categories")
    parser.add_argument("-mqps", "--max_q_per_subject", type = int, default = None, help = "Max questions per subject (default: unlimited)")
    parser.add_argument("-r", "--random_sample", type = int, default = None, help = "Randomly sample number of questions across subjects")
    _args = parser.parse_args()
    main(_args)
