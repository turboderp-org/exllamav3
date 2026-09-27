# Building a frozen draft corpus

From the repository root, with ExLlamaV3 installed:

```bash
pip install -r requirements_sam.txt
python util/build_sam.py \
    -m /path/to/model \
    -r util/sam_recipes/coding.yaml \
    -o coding.sam.zst
```

`-m` is a local directory containing the tokenizer and chat template; weights are
not loaded. The recipe above is also the default. Use `--target-bytes 1000000`
for a small trial. `--max-tokens` overrides the hard token limit;
`--compression-level` defaults to 9. Existing outputs require `--force`.
A failed build leaves an existing output intact. A completed output is published
atomically. The builder needs the extension containing `BC_SAM.export_csr`.

This builds **one frozen bank**. An artifact belongs to its tokenizer and
chat-rendering configuration, not just its model architecture.

Use it in the speculative decoding benchmark:

```bash
python eval/spec_decode.py -m /path/to/model -single 'Agentic*' \
    -ngram_min 2 -ngram_len 15 -ngram_corpus coding.sam.zst
```

Or pass `ngram_corpus="coding.sam.zst"` to `Generator`, together with
`ngram_match_min=2` and `num_draft_tokens=15`. Install `zstandard` in the runtime
environment. The generator decompresses and validates the bank once, checks
tokenizer file identities, and shares its immutable storage between independent
job cursors. Each job still builds its live history SAM. Drafting uses the longest
match that supplies a continuation, prefers live history on equal match lengths,
and stops corpus continuations at document boundaries. Loading is outside the
benchmark's generation timing. The runtime uses byte planes directly; it does
not rebuild the graph. Different quantization/model configs and library versions
are allowed, but tokenizer.json, tokenizer_config.json and added_tokens.json
identities must match. Inference template options may differ, which can affect
corpus coverage but does not change token IDs.

## Default mixture and sampling

The default recipe targets 100 MB of rendered UTF-8 text:

| Source | Share | Material |
| --- | ---: | --- |
| SWE-smith trajectories | 40% | Resolved tool conversations, rendered with the model's chat template |
| Magicoder OSS-Instruct | 30% | Solution text/code |
| OpenCoder package_instruct | 20% | Output text/code |
| SmolTalk smol-magpie-ultra | 10% | Assistant response bodies |

These are starting proportions to evaluate with a speculative decoding benchmark.
SWE-smith includes user messages and tool observations as well as assistant turns;
keeping whole conversations allows templates to serialize tool calls correctly.
The other sources are response bodies encoded without additional special tokens.
No template is applied to plain text/code. Tool names and arguments in the corpus
remain those used by the source; the builder does not translate them to another
agent framework's tool API.

All sources use HF `datasets` streaming. Parquet sources project only the recipe's
`columns`; notably, SWE-smith's patch column is not requested. Magicoder's original
JSON source is streamed without column projection. A small bounded shuffle buffer
(default 32 records) provides local variety. This is **not a uniform sample of the
entire dataset**. Data order can change across datasets-library versions, which is
why the artifact also records the actual corpus hash and dependency versions.

Streaming avoids materializing complete datasets, but metadata, compressed blocks,
row groups and read-ahead still incur downloads. The byte target is **not a network
transfer cap**. Authentication and caches use the standard HF environment settings
(e.g. `HF_TOKEN`, `HF_HOME`). Recipe revisions are pinned; resolved commit IDs are
also saved in the artifact.

Sampling stops at each source's target or `max_rows`. Whole documents are retained,
so each source can overshoot its target by at most `max_document_bytes` (512 KB by
default). Oversized, empty, duplicate and malformed documents are skipped and
counted. The first three normalization errors per source are reported. A source
with no usable documents fails the build; an underfilled source reports
`target_reached: false` and does not silently redistribute its budget. Complete
rendered documents are deduplicated across all sources. Filters apply before
rendering; they are equality checks, not Python expressions.

Sampling/tokenization is single-threaded at the Python level, as is SAM building.
The tokenizer library may use its own internal workers. Building needs substantially
more RAM than the final compressed artifact; allow several GB for a 100 MB text
recipe. The live SAM, frozen arrays and token IDs overlap briefly in memory. The
hard token limit includes document separators and prevents int32 index overflow;
it is not a RAM limit. Source sequence lengths are not clipped to the model's
attention context window: this bank is an offline text index.

For reference, the default recipe with the Gemma 4 tokenizer on a Threadripper
7960X produced 29,083,418 tokens from 100.1 MB of text: a 314 MB compressed bank,
48.7 s sampling/tokenization, 18.8 s building/freezing, 12.3 s compression/write,
and 4.3 GiB peak process memory. Network/cache state and tokenizer choice affect
these figures. This validates the build, not speculative decoding acceptance.

## Recipe fields

Recipes are YAML mappings with `version: 1`, a `sources` list, and optional
`target_bytes`, `max_tokens`, `max_document_bytes`, `seed`, `shuffle_buffer`, and
`chat_template_kwargs`. Set the latter to match inference (for example,
`{enable_thinking: false}` if the template supports it). `add_generation_prompt`
is always false, and templates render complete conversations.

Each source requires `name`, `dataset` (HF dataset repository), `split`, `weight`
(a positive integer), `mode`, and `field`. Optional fields are `config`, `revision`,
`max_rows` (default 50,000), `columns` (Parquet projection), `filters` (field/value
mapping), and `tools_field` (row field containing the tool schema list).

- `text`: `field` contains a string; tokenize its body directly.
- `assistant_text`: `field` contains messages or a JSON message string; each
  assistant's text body becomes a separate document.
- `chat`: normalize messages and render the entire conversation through the
  model's Jinja template. OpenAI-style tool argument JSON strings become objects;
  text content blocks become strings. SWE-smith's single `tool_call_ids` entry is
  converted to `tool_call_id`, and tool names are resolved from preceding calls.

Unsupported content (including images) and ambiguous tool responses are rejected
for that record, not silently flattened. Custom dataset loading scripts and model
remote code are not enabled. A template exception aborts the build unless it is a
normalization/value error recorded by the sampler.

## Frozen format, version 1

All offsets below address the **decompressed payload**, not the file. The file is:

1. A 24-byte little-endian prefix: `8s magic` (`EXL3SAM\0`), `uint32 version` (1),
   `uint32 JSON_length`, `uint64 payload_length`.
2. Exactly `JSON_length` bytes of UTF-8 JSON metadata.
3. One standard Zstandard frame with pledged content size and frame checksum.

Metadata describes every section (name, offset, count, dtype, encoding), graph
sizes, tokenizer/config/template file hashes and effective template, dependency
versions, effective recipe, resolved source revisions and sampling statistics.
`corpus_sha256` hashes the little-endian int32 corpus, including separators.
The tokenizer identity hash covers its files, template, rendering kwargs, class
and tokenizer library versions. The loader checks tokenizer compatibility and
validates sizes/indices before exposing graph arrays.

Sections start at 64-byte-aligned payload offsets. They contain signed little-endian
int32 values in **byte-plane order**: all first bytes, then all second bytes, then
all third bytes, then all fourth bytes. Thus word `i` of an `N`-word section is
`b[i] | b[N+i]<<8 | b[2*N+i]<<16 | b[3*N+i]<<24`, interpreted as signed int32.
A reader can either access those planes directly or unshuffle once on loading.
Padding is zero. There are no pointers or serialized hash maps.

| Section | Count | Meaning |
| --- | ---: | --- |
| `link` | states | Suffix links, root = -1 |
| `max_len` | states | Maximum represented substring length |
| `min_end` | states | Representative occurrence's inclusive corpus end; root is INT32_MAX |
| `edge_offsets` | states + 1 | CSR edge ranges; final offset = edges |
| `edge_token` | edges | Transition labels, sorted by signed token ID within each state |
| `edge_to` | edges | Destination state IDs |
| `root` | variable | Dense root destination IDs, missing = -1 |
| `corpus` | tokens | Token history, including -1 document separators |
| `document_ends` | documents | Exclusive document ends, pointing to separator positions |

The graph is the SAM of the joined corpus. Real tokenizer IDs must be nonnegative;
-1 is reserved and cannot occur in a model query, so matches cannot span documents.
The drafter **clamps continuations at the next document end** and
never returns a separator. Root transitions with IDs outside the dense table use
the root's CSR range; `root_dense_limit` is 1,048,576. Nonroot CSR lookups can use
linear scans for small degrees and binary search for larger ones. Matching cursor
state is per session and is not stored in the file.
