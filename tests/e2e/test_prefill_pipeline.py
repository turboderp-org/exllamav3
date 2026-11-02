"""
Prefill across a layer split with the devices working on different chunks at once (PrefillPipeline) changes the order
in which the devices do their work and nothing else, so its output has to match the serial path's exactly: the same
kernels run on the same inputs. That goes for the logits after the prompt, and for those after restoring from the
checkpoints written behind chunks in flight, at the end of the reply and from inside the prompt, with and without a
drafter sharing a device with the target, and for image jobs with the image across a chunk boundary, where Gemma4
extends the chunk over the image and rewinds the state after the pass, and DeepSeek-V4 cuts its chunks to the image.

Each model is split across two devices by weight size; a split the pipeline declines is skipped.
"""

import glob
import os

import pytest
import torch

from exllamav3 import Model
from PIL import Image
from testlib.e2e import REFERENCE_TEXT, greedy, load_model
from testlib.env import num_devices

pytestmark = [pytest.mark.multi_gpu(2), pytest.mark.slow]

PAGE = 256
CHUNK = 4 * PAGE
NEW_TOKENS = 16
REPLY_TOKENS = PAGE + 60


def _split(model_dir: str, devices: list[torch.device]) -> str:
    """--gpu_split value putting about half of the weights on the first device and the rest on the second"""
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(model_dir, "*.safetensors"))) / 1024**3
    split = ["0"] * num_devices()
    split[devices[0].index] = f"{max(size * 0.55, 0.25):.2f}"
    split[devices[1].index] = "1000"
    return ",".join(split)


def _distinct_hash_routing(lm):
    """Give a stub's hash-routed layers distinct experts per token, as a real checkpoint's tid2eid has. Stubs generated
    with repeats add one expert twice for a token, and the racing additions make the serial path vary from run to run"""
    g = torch.Generator().manual_seed(0)
    for m in lm.model:
        table = getattr(getattr(m, "routing_cfg", None), "tid2eid", None)
        if table is not None:
            rows = torch.rand(table.shape[0], m.num_experts, generator = g).argsort(-1)
            with torch.inference_mode():
                table.copy_(rows[:, : table.shape[1]].to(table.device, table.dtype))


def _conversation(lm, pipelined: bool, ids: torch.Tensor = None, embeddings: list = None) -> list:
    gen = lm.generator(pipelined_prefill = pipelined, max_chunk_size = CHUNK)
    if pipelined and gen.pipeline is None:
        gen.close()
        pytest.skip("the pipeline declined this split")
    if ids is None:
        ids = lm.tokenizer.encode(REFERENCE_TEXT * 40, add_bos = True)[:, : 5 * CHUNK + 100]
    assert ids.shape[-1] > 4 * CHUNK
    job = dict(return_logits = True, embeddings = embeddings)
    turns = [greedy(gen, ids, REPLY_TOKENS, **job)]
    # Restores the checkpoint at the end of the reply
    suffix = lm.tokenizer.encode("\n\nAnd then?\n", add_bos = False)
    followup = torch.cat((ids, torch.tensor([turns[0][0]]), suffix), dim = -1)
    turns.append(greedy(gen, followup, NEW_TOKENS, **job))
    # Restores one from inside the prompt, written behind the chunks then in flight
    turns.append(greedy(gen, ids[:, : -CHUNK], NEW_TOKENS, **job))
    gen.close()
    return turns


def _assert_exact(serial: list, pipelined: list):
    for i, ((ref_tokens, ref_logits, ref_events), (tokens, logits, events)) in enumerate(zip(serial, pipelined)):
        assert tokens == ref_tokens, f"turn {i}: tokens differ"
        finite = torch.isfinite(ref_logits).all(0) & torch.isfinite(logits).all(0)
        assert torch.equal(logits[:, finite], ref_logits[:, finite]), f"turn {i}: logits differ"
        assert events["eos"]["cached_tokens"] == ref_events["eos"]["cached_tokens"], f"turn {i}: resumed elsewhere"


@pytest.mark.models()
def test_pipelined_prefill_is_exact(model_id, model_dir, devices):
    with load_model(model_dir, None, "-gs", _split(model_dir, devices[:2]), cache_tokens = 64 * PAGE) as lm:
        _distinct_hash_routing(lm)
        pipelined = _conversation(lm, True)
        serial = _conversation(lm, False)
    _assert_exact(serial, pipelined)


@pytest.mark.models("vision")
def test_pipelined_prefill_with_an_image_is_exact(model_id, model_dir, devices):
    with load_model(model_dir, None, "-gs", _split(model_dir, devices[:2]), cache_tokens = 64 * PAGE) as lm:
        _distinct_hash_routing(lm)
        vision = Model.from_config(lm.config, component = "vision")
        vision.load(device = devices[0])
        media = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "media", "cat.png")
        image = vision.get_image_embeddings(tokenizer = lm.tokenizer, image = Image.open(media).convert("RGB"))
        vision.unload()
        # Ending just before a chunk boundary, the image's tokens run across it
        text = lm.tokenizer.encode(REFERENCE_TEXT * 40, add_bos = False)
        head = lm.tokenizer.decode(text[:, : CHUNK - 40])[0]
        tail = lm.tokenizer.decode(text[:, CHUNK - 40 : 5 * CHUNK])[0]
        ids = lm.tokenizer.encode(f"{head}\n{image.text_alias}\n{tail}", add_bos = True, encode_special_tokens = True,
                                  embeddings = [image])[:, : 5 * CHUNK + 100]
        pipelined = _conversation(lm, True, ids, [image])
        serial = _conversation(lm, False, ids, [image])
    _assert_exact(serial, pipelined)


@pytest.mark.model()
@pytest.mark.parametrize("role, drafting", [
    ("mtp", lambda dirs: ("-mtp",)),
    ("dflash2-target", lambda dirs: ("-dm", dirs["draft"])),
], ids = ["mtp", "dflash2"])
def test_pipelined_prefill_with_a_drafter_is_exact(role, drafting, model_registry, devices):
    entry = model_registry.get(role)
    if not entry.available:
        pytest.skip(f"test model '{role}' not available")
    if entry.draft and not model_registry.get(entry.draft).available:
        pytest.skip(f"draft model '{entry.draft}' not available")
    dirs = {"draft": model_registry.get(entry.draft).path if entry.draft else None}
    with load_model(entry.path, None, "-gs", _split(entry.path, devices[:2]), *drafting(dirs), cache_tokens = 64 * PAGE) as lm:
        pipelined = _conversation(lm, True)
        serial = _conversation(lm, False)
    _assert_exact(serial, pipelined)
    assert pipelined[0][2]["eos"]["accepted_draft_tokens"] == serial[0][2]["eos"]["accepted_draft_tokens"]
