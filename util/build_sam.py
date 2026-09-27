#!/usr/bin/env python3
"""Build one model-specific, compressed frozen suffix-automaton corpus.

See build_sam.md for recipes and the versioned file format. No model weights
are loaded. Hugging Face sources are streamed, never materialized in full.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

MAGIC = b'EXL3SAM\0'
VERSION = 1
PREFIX = struct.Struct('<8sIIQ')  # magic, version, JSON bytes, decoded payload bytes
GRAPH_ARRAYS = ('link', 'max_len', 'min_end', 'edge_offsets', 'edge_token', 'edge_to', 'root')
MAX_TOKENS = (2**31 - 2) // 3  # conservative state/edge index bound, including separators
SEPARATOR = -1


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def load_recipe(path):
    import yaml
    recipe = yaml.safe_load(Path(path).read_text())
    if not isinstance(recipe, dict) or recipe.get('version') != 1:
        raise ValueError('Recipe must be a mapping with version: 1')
    allowed = {'version', 'target_bytes', 'seed', 'shuffle_buffer', 'max_document_bytes',
               'max_tokens', 'chat_template_kwargs', 'sources'}
    if extra := recipe.keys() - allowed:
        raise ValueError(f'Unknown recipe fields: {sorted(extra)}')
    for k, default in [('target_bytes', 100_000_000), ('shuffle_buffer', 32),
                       ('max_document_bytes', 512_000), ('max_tokens', 80_000_000)]:
        recipe[k] = positive(recipe.get(k, default), k)
    if recipe['max_tokens'] > MAX_TOKENS:
        raise ValueError(f'max_tokens cannot exceed {MAX_TOKENS}')
    recipe.setdefault('seed', 42)
    if type(recipe['seed']) is not int:
        raise ValueError('seed must be an integer')
    kwargs = recipe.setdefault('chat_template_kwargs', {})
    if not isinstance(kwargs, dict) or kwargs.keys() & {'tokenize', 'return_dict', 'add_generation_prompt', 'tools', 'conversation'}:
        raise ValueError('Invalid or reserved chat_template_kwargs')
    sources = recipe.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Recipe needs a nonempty sources list')
    names = set()
    for source in sources:
        allowed = {'name', 'dataset', 'config', 'split', 'revision', 'columns', 'weight',
                   'max_rows', 'mode', 'field', 'tools_field', 'filters'}
        if not isinstance(source, dict) or source.keys() - allowed:
            raise ValueError(f'Invalid source fields: {source}')
        for key in ('name', 'dataset', 'split', 'field'):
            if not isinstance(source.get(key), str) or not source[key]:
                raise ValueError(f'Source requires string {key}')
        if source['name'] in names:
            raise ValueError(f'Duplicate source name: {source["name"]}')
        names.add(source['name'])
        positive(source.get('weight'), 'source weight')
        source['max_rows'] = positive(source.get('max_rows', 50_000), 'max_rows')
        if source.get('mode') not in ('text', 'assistant_text', 'chat'):
            raise ValueError('mode must be text, assistant_text or chat')
        if not isinstance(source.get('filters', {}), dict):
            raise ValueError('filters must be a field: value mapping')
        if 'columns' in source:
            columns = source['columns']
            needed = {source['field'], *source.get('filters', {})}
            if source.get('tools_field'):
                needed.add(source['tools_field'])
            if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns) or not needed <= set(columns):
                raise ValueError(f'columns must include {sorted(needed)}')
    return recipe


def text_content(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, dict) and x.get('type') == 'text' and isinstance(x.get('text'), str) for x in value):
        return '\n'.join(x['text'] for x in value)
    raise ValueError('Expected text-only message content')


def normalize_messages(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError('Expected a nonempty message list')
    result = []
    tool_names = {}
    for item in value:
        if not isinstance(item, dict) or item.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
            raise ValueError('Unsupported message role')
        msg = {'role': item['role'], 'content': text_content(item.get('content'))}
        for key in ('name', 'tool_call_id', 'reasoning_content'):
            if key in item:
                msg[key] = item[key]
        if item.get('tool_calls'):
            calls = []
            for call in item['tool_calls']:
                function = call['function']
                args = function['arguments']
                if isinstance(args, str):
                    args = json.loads(args)
                if not isinstance(args, dict) or not isinstance(function['name'], str):
                    raise ValueError('Tool arguments must decode to an object')
                normalized = {'type': 'function', 'function': {'name': function['name'], 'arguments': args}}
                if 'id' in call:
                    normalized['id'] = call['id']
                    tool_names[call['id']] = function['name']
                calls.append(normalized)
            msg['tool_calls'] = calls
        ids = item.get('tool_call_ids')
        if ids:
            if len(ids) != 1:
                raise ValueError('Cannot assign one tool response to multiple calls')
            msg['tool_call_id'] = ids[0]
        if msg['role'] == 'tool' and msg.get('tool_call_id') in tool_names:
            msg.setdefault('name', tool_names[msg['tool_call_id']])
        result.append(msg)
    return result


def documents(row, source, tokenizer, template_kwargs):
    value = row[source['field']]
    if source['mode'] == 'text':
        if not isinstance(value, str):
            raise ValueError('Text field is not a string')
        return [value]
    messages = normalize_messages(value)
    if source['mode'] == 'assistant_text':
        return [m['content'] for m in messages if m['role'] == 'assistant' and m['content']]
    tools = row[source['tools_field']] if source.get('tools_field') else None
    if isinstance(tools, str):
        tools = json.loads(tools)
    return [tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                          add_generation_prompt=False, **template_kwargs)]


def stream_source(source, seed, buffer_size):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(source['dataset'], revision=source.get('revision')).sha
    options = dict(path=source['dataset'], name=source.get('config'), split=source['split'],
                   revision=revision, streaming=True)
    if source.get('columns'):
        options['columns'] = source['columns']
    dataset = load_dataset(**options)
    # A bounded local shuffle, not an expensive uniform sample of the full corpus.
    if buffer_size > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return iter(dataset), revision


def collect(source, rows, tokenizer, kwargs, budget, max_document_bytes,
            max_tokens, tokens, ends, seen):
    stats = dict(rows=0, documents=0, bytes=0, tokens=0, filtered=0,
                 duplicates=0, oversized=0, invalid=0, empty=0, errors=[])
    # Check the bound before requesting the next row (important for remote streams).
    while stats['rows'] < source['max_rows'] and stats['bytes'] < budget:
        try:
            row = next(rows)
        except StopIteration:
            break
        stats['rows'] += 1
        if any(row.get(k) != v for k, v in source.get('filters', {}).items()):
            stats['filtered'] += 1
            continue
        try:
            docs = documents(row, source, tokenizer, kwargs)
        except (ValueError, KeyError, TypeError) as exc:
            stats['invalid'] += 1
            if len(stats['errors']) < 3:
                stats['errors'].append(f'{type(exc).__name__}: {exc}')
            continue
        for doc in docs:
            if stats['bytes'] >= budget:
                break
            encoded = doc.encode('utf-8')
            if not encoded.strip():
                stats['empty'] += 1
                continue
            if len(encoded) > max_document_bytes:
                stats['oversized'] += 1
                continue
            digest = hashlib.sha256(encoded).digest()
            if digest in seen:
                stats['duplicates'] += 1
                continue
            ids = tokenizer.encode(doc, add_special_tokens=False)
            if not ids:
                stats['empty'] += 1
                continue
            if any(type(t) is not int or t < 0 or t > 2**31 - 1 for t in ids):
                raise ValueError('Tokenizer produced IDs outside the nonnegative int32 range')
            if len(tokens) + len(ids) + 1 > max_tokens:
                raise ValueError('Token limit exceeded; reduce target_bytes or increase max_tokens')
            tokens.extend(ids)
            ends.append(len(tokens))  # exclusive end, before the separator
            tokens.append(SEPARATOR)
            seen.add(digest)
            stats['documents'] += 1
            stats['bytes'] += len(encoded)
            stats['tokens'] += len(ids)
    stats['target_bytes'] = budget
    stats['target_reached'] = stats['bytes'] >= budget
    return stats


def tokenizer_metadata(model_dir, tokenizer, kwargs):
    import transformers
    import tokenizers
    # Hash tokenizer/config/template files, never model weights.
    paths = [p for p in model_dir.rglob('*') if p.is_file() and
             (p.suffix == '.jinja' or p.name in ('tokenizer.json', 'tokenizer_config.json',
              'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model',
              'spiece.model', 'vocab.json', 'vocab.txt', 'merges.txt', 'config.json'))]
    files = {}
    for path in sorted(paths):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        files[str(path.relative_to(model_dir))] = h.hexdigest()
    identity = dict(files=files, tokenizer_class=type(tokenizer).__name__,
                    chat_template=tokenizer.chat_template, chat_template_kwargs=kwargs,
                    transformers=transformers.__version__, tokenizers=tokenizers.__version__)
    return dict(sha256=hashlib.sha256(canonical(identity)).hexdigest(), **identity)


def write_bank(output, arrays, metadata, level=9, force=False):
    """Atomic writer: little-endian byte planes, aligned arrays, one Zstd frame."""
    import numpy as np
    import zstandard as zstd
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(output)
    sections = []
    cursor = 0
    for name, values in arrays.items():
        if values.ndim != 1 or values.dtype.kind != 'i' or values.dtype.itemsize != 4:
            raise ValueError(f'{name} must be a 1D int32 array')
        cursor = (cursor + 63) & ~63
        sections.append(dict(name=name, offset=cursor, count=len(values), dtype='<i4', encoding='byte_planes'))
        cursor += len(values) * 4
    meta = dict(metadata, format='exllamav3.frozen_sam', version=VERSION,
                layout='csr', compression='zstd', sections=sections, payload_bytes=cursor,
                separator=SEPARATOR, root_dense_limit=1 << 20)
    header = canonical(meta)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + '.', suffix='.tmp', delete=False) as f:
            temp = Path(f.name)
            f.write(PREFIX.pack(MAGIC, VERSION, len(header), cursor))
            f.write(header)
            with zstd.ZstdCompressor(level=level, write_checksum=True).stream_writer(f, size=cursor, closefd=False) as writer:
                at = 0
                for section, values in zip(sections, arrays.values()):
                    writer.write(bytes(section['offset'] - at))
                    words = np.asarray(values, dtype='<i4').view(np.uint8).reshape(-1, 4)
                    for plane in range(4):
                        for start in range(0, len(words), 1024 * 1024):
                            writer.write(words[start:start + 1024 * 1024, plane].tobytes())
                    at = section['offset'] + section['count'] * 4
            f.flush()
            os.fsync(f.fileno())
        if force:
            os.replace(temp, output)
        else:
            # Atomic no-clobber publication, including a race with another builder.
            os.link(temp, output)
            temp.unlink()
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model-dir', type=Path, required=True)
    parser.add_argument('-r', '--recipe', type=Path, default=Path(__file__).with_name('sam_recipes') / 'coding.yaml')
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--target-bytes', type=int, help='Override total rendered UTF-8 byte target')
    parser.add_argument('--max-tokens', type=int, help='Hard token limit, including document separators')
    parser.add_argument('--compression-level', type=int, default=9, choices=range(1, 20), metavar='1..19')
    parser.add_argument('--force', action='store_true', help='Replace an existing output')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    for name in ('target_bytes', 'max_tokens'):
        if (value := getattr(args, name)) is not None:
            recipe[name] = positive(value, name)
    if recipe['max_tokens'] > MAX_TOKENS:
        raise ValueError(f'max_tokens cannot exceed {MAX_TOKENS}')
    if not args.model_dir.is_dir():
        raise ValueError('model-dir must be a local model directory')
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise ValueError('Output directory does not exist')
    import numpy as np
    import torch
    import zstandard  # fail before downloading if the optional dependency is missing
    from transformers import AutoTokenizer
    # Support both `python util/build_sam.py` and module invocation from a checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from exllamav3.ext import exllamav3_ext as ext
    if not hasattr(ext.BC_SAM, 'export_csr'):
        raise RuntimeError('Rebuild the exllamav3 extension: BC_SAM.export_csr is missing')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError('Model tokenizer has no chat template')
    identity = tokenizer_metadata(args.model_dir, tokenizer, recipe['chat_template_kwargs'])
    tokens, ends = array('i'), array('i')
    if tokens.itemsize != 4:
        raise RuntimeError('This platform does not have 32-bit C int arrays')
    seen, reports = set(), []
    start = time.perf_counter()
    total_weight = sum(s['weight'] for s in recipe['sources'])
    allocated = 0
    for i, source in enumerate(recipe['sources']):
        quota = recipe['target_bytes'] * source['weight'] // total_weight
        if i == len(recipe['sources']) - 1:
            quota = recipe['target_bytes'] - allocated
        allocated += quota
        if quota == 0:
            continue
        print(f"Sampling {source['name']} (target {quota:,} bytes)...", flush=True)
        rows, revision = stream_source(source, recipe['seed'] + i, recipe['shuffle_buffer'])
        stats = collect(source, rows, tokenizer, recipe['chat_template_kwargs'], quota,
                        recipe['max_document_bytes'], recipe['max_tokens'], tokens, ends, seen)
        del rows
        print(json.dumps(stats, ensure_ascii=False), flush=True)
        if not stats['documents']:
            raise ValueError(f"No usable documents from {source['name']}; check recipe/errors above")
        reports.append(dict(name=source['name'], revision=revision, **stats))
    if not tokens:
        raise ValueError('No corpus tokens collected')
    del seen
    ids = np.frombuffer(tokens, dtype=np.int32)
    corpus_hash = hashlib.sha256(ids.astype('<i4', copy=False).tobytes()).hexdigest()
    sampled = time.perf_counter()
    print(f'Building single SAM: {len(ids):,} tokens, {len(ends):,} documents...', flush=True)
    sam = ext.BC_SAM()
    sam.reset(len(ids))
    history = torch.from_numpy(ids).to(dtype=torch.int64)
    sam.accept_tensor(history)
    del history
    graph = sam.export_csr()
    del sam
    built = time.perf_counter()
    arrays = dict(zip(GRAPH_ARRAYS, (tensor.numpy() for tensor in graph)))
    arrays['corpus'] = ids
    arrays['document_ends'] = np.frombuffer(ends, dtype=np.int32)
    metadata = dict(tokenizer=identity, recipe=recipe, sources=reports,
                    software={name: package_version(name) for name in
                              ('datasets', 'huggingface-hub', 'numpy', 'torch', 'transformers', 'tokenizers', 'zstandard')},
                    corpus_sha256=corpus_hash, token_count=len(ids), document_count=len(ends),
                    state_count=len(arrays['link']), edge_count=len(arrays['edge_token']))
    write_bank(args.output, arrays, metadata, args.compression_level, args.force)
    done = time.perf_counter()
    print(f'Wrote {args.output}: {args.output.stat().st_size:,} bytes; '
          f'sampling/tokenization {sampled-start:.2f}s, build/freeze {built-sampled:.2f}s, '
          f'compression/write {done-built:.2f}s', flush=True)


if __name__ == '__main__':
    main()
