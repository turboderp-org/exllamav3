import torch
import os
import random

def split_art(articles, rows, columns, tokenizer):
    t_rows = []
    idx = 0
    empty = torch.empty((1, 0), dtype = torch.long)
    t_row = empty
    while len(t_rows) < rows:
        add_special_tokens = (len(t_rows) % 2 == 0)
        t_art = tokenizer.encode(articles[idx], add_bos = add_special_tokens, add_eos = add_special_tokens)
        t_row = torch.cat((t_row, t_art), dim = -1)
        t_row = t_row[:, :columns]
        if t_row.shape[-1] == columns:
            t_rows.append(t_row)
            t_row = empty
        idx += 1
    return t_rows


def split_wiki(text, rows, columns, tokenizer):
    articles = [a[a.find("\n") + 1:] for a in text.split("</doc>\n")]
    articles = [a for a in articles if len(a) > 50]
    return split_art(articles, rows, columns, tokenizer)


def split_tiny(text, rows, columns, tokenizer):
    articles = [a.strip() for a in text.split("<|endoftext|>")]
    return split_art(articles, rows, columns, tokenizer)


def shuffle_lines(text, rows, columns, tokenizer):
    articles = text.split("\n")
    articles = [a for a in articles if not a.isspace()]
    random.seed(0)
    random.shuffle(articles)
    return split_art(articles, rows, columns, tokenizer)


def split_raw(text, rows, columns, tokenizer):
    t_all = tokenizer.encode(text)
    t_rows = []
    for i in range(rows):
        a = i * columns
        b = a + columns
        t_rows.append(t_all[:, a:b])
    return t_rows


def random_data(text, rows, columns, tokenizer):
    vocab_size = tokenizer.actual_vocab_size
    torch.manual_seed(0)
    t_rows = []
    for i in range(rows):
        t_row = torch.randint(0, vocab_size, (1, columns), dtype = torch.long)
        t_rows.append(t_row)
    return t_rows


def get_default_calibration(args, tokenizer):
    columns = args["cal_cols"]
    rows = args["cal_rows"]

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_cal_data")
    files = [
        ("c4.utf8", 20, shuffle_lines),
        ("code.utf8", 20, split_raw),
        ("multilingual.utf8", 10, shuffle_lines),
        ("technical.utf8", 10, split_raw),
        ("wiki.utf8", 50, split_wiki),
        ("tiny.utf8", 5, split_tiny),
        (None, 20, random_data),
    ]

    dist_sum = sum(x for (_, x, _) in files)
    cal_data = []

    for filename, weight, processor in files:
        target_rows = max(1, int(weight / dist_sum * rows))
        if filename:
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding = "utf8") as f:
                file_text = f.read()
        else:
            file_text = None
            target_rows = max(1, rows - len(cal_data))
        r = processor(file_text, target_rows, columns, tokenizer)
        cal_data += r

    # cal_data = torch.cat(cal_data, dim = 0)
    return cal_data