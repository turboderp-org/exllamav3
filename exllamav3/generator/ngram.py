"""Read-only frozen SAM banks shared by independent per-job matching cursors."""
import hashlib
import json
from pathlib import Path
import struct
import time
import logging

import numpy as np
import torch
from ..ext import exllamav3_ext as ext

logger = logging.getLogger(__name__)
_NAMES = ('link', 'max_len', 'min_end', 'edge_offsets', 'edge_token', 'edge_to', 'root', 'corpus', 'document_ends')
_PREFIX = struct.Struct('<8sIIQ')


class NgramCorpus:
    def __init__(self, path, tokenizer):
        try:
            import zstandard
        except ImportError as exc:
            raise ImportError('Frozen SAM loading requires zstandard: pip install zstandard') from exc
        start = time.perf_counter()
        with open(path, 'rb') as f:
            prefix = f.read(_PREFIX.size)
            if len(prefix) != _PREFIX.size:
                raise ValueError('Truncated frozen SAM header')
            magic, version, header_size, size = _PREFIX.unpack(prefix)
            if magic != b'EXL3SAM\0' or version != 1 or header_size > 16 * 1024**2:
                raise ValueError('Unsupported frozen SAM header')
            self.metadata = m = json.loads(f.read(header_size))
            if m.get('format') != 'exllamav3.frozen_sam' or m.get('version') != 1 or m.get('layout') != 'csr' or m.get('compression') != 'zstd' or m.get('separator') != -1 or m.get('payload_bytes') != size:
                raise ValueError('Unsupported frozen SAM layout')
            self.check_tokenizer(tokenizer)
            sections = m.get('sections', [])
            if [s.get('name') for s in sections] != list(_NAMES):
                raise ValueError('Invalid frozen SAM sections')
            end = 0
            for section in sections:
                offset, count = section['offset'], section['count']
                if type(offset) is not int or type(count) is not int or count < 0 or offset < end or offset % 64 or section.get('dtype') != '<i4' or section.get('encoding') != 'byte_planes':
                    raise ValueError('Invalid frozen SAM section descriptor')
                end = offset + count * 4
            if end != size or size == 0:
                raise ValueError('Invalid frozen SAM payload size')
            # Decode directly into writable backing storage, with no unshuffle pass.
            self.storage = bytearray(size)
            view = memoryview(self.storage)
            with zstandard.ZstdDecompressor().stream_reader(f) as reader:
                at = 0
                while at < size:
                    n = reader.readinto(view[at:min(at + 16 * 1024**2, size)])
                    if not n:
                        raise ValueError('Truncated frozen SAM payload')
                    at += n
                if reader.read(1):
                    raise ValueError('Unexpected trailing frozen SAM data')
        self.sections = [torch.frombuffer(view[s['offset']:s['offset'] + s['count'] * 4], dtype=torch.uint8)
                         if s['count'] else torch.empty(0, dtype=torch.uint8) for s in sections]
        self.vocab_size = tokenizer.actual_vocab_size
        self._validate()
        logger.info('Loaded frozen SAM %s (%s tokens) in %.2fs', path, m['token_count'], time.perf_counter()-start)

    def check_tokenizer(self, tokenizer):
        files = self.metadata.get('tokenizer', {}).get('files', {})
        directory = Path(tokenizer.config.directory)
        # Model config/quantization and library versions may differ. Token IDs may not.
        if 'tokenizer.json' not in files:
            raise ValueError('Corpus lacks a tokenizer.json identity')
        for name in ('tokenizer.json', 'tokenizer_config.json', 'added_tokens.json'):
            path = directory / name
            expected = files.get(name)
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if expected != actual:
                raise ValueError(f'Frozen SAM tokenizer mismatch: {name}')

    def _validate(self):
        def unpack(i):
            return self.sections[i].numpy().reshape(4, -1).T.copy().view('<i4').ravel()
        def require(condition):
            if not condition:
                raise ValueError('Invalid frozen SAM graph')
        m = self.metadata
        n, edges, tokens = m['state_count'], m['edge_count'], m['token_count']
        require(n > 0 and tokens > 0)
        require([t.numel() // 4 for t in self.sections] ==
                [n, n, n, n+1, edges, edges, self.sections[6].numel()//4, tokens, m['document_count']])
        links, lengths = unpack(0), unpack(1)
        require(links[0] == -1 and lengths[0] == 0)
        require(np.all((links[1:] >= 0) & (links[1:] < n)))
        require(np.all(lengths[1:] > lengths[links[1:]]) and lengths.max() <= tokens)
        ends = unpack(2)
        require(np.all((ends[1:] >= lengths[1:] - 1) & (ends[1:] < tokens)))
        del links, lengths, ends
        offsets = unpack(3)
        require(offsets[0] == 0 and offsets[-1] == edges and np.all(offsets[1:] >= offsets[:-1]))
        dest = unpack(5)
        require(np.all((dest >= 0) & (dest < n)))
        labels = unpack(4)
        # Non-increasing labels are allowed only at state boundaries.
        breaks = np.flatnonzero(labels[1:] <= labels[:-1]) + 1
        require(np.all(np.isin(breaks, offsets, assume_unique=False)))
        root = unpack(6)
        require(len(root) <= 1 << 20 and np.all((root >= -1) & (root < n)))
        expected = np.full(len(root), -1, dtype=np.int32)
        a, b = offsets[:2]
        root_labels = labels[a:b]
        mask = (root_labels >= 0) & (root_labels < len(root))
        expected[root_labels[mask]] = dest[a:b][mask]
        require(np.array_equal(root, expected))
        del offsets, dest, labels, breaks, root, expected, root_labels
        corpus, boundaries = unpack(7), unpack(8)
        require(hashlib.sha256(memoryview(corpus)).hexdigest() == m['corpus_sha256'])
        require(len(boundaries) > 0 and boundaries[-1] == tokens-1 and boundaries[0] >= 0)
        require(np.all(boundaries[1:] > boundaries[:-1]))
        require(np.all((corpus >= -1) & (corpus < self.vocab_size)) and np.all(corpus[boundaries] == -1) and np.count_nonzero(corpus == -1) == len(boundaries))

    def cursor(self):
        return ext.FrozenSAMCursor(self.sections)
