from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import model_init
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent
app = FastAPI()

class Session:

    def __init__(self, model, config, cache, tokenizer):
        self.model = model
        self.config = config
        self.cache = cache
        self.tokenizer = tokenizer
        self.eos_token_ids = config.eos_token_id_list or []
        self.prompt = self.default_prompt()

        self.tokens: list[int] = []             # flat committed token ids
        self.blocks: list[dict] = []            # [{tokens, text, is_prompt}]
        self.cache_len: int = 0                 # how much of cache we logically trust
        self.pending_branches: list[dict] = []  # [{tokens, text, prob}]

        self.last_logits: torch.Tensor | None = None


    def default_prompt(self):
        return self.model.default_chat_prompt("Hello.", "You are a super helpful AI assistant.")


    def _forward_one(self, token_id: int, position: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], dtype = torch.long)
        params = {
            "attn_mode": "flash_attn",
            "cache": self.cache,
            "past_len": position,
            "batch_shape": (1, self.cache.max_num_tokens),
        }
        return self.model.forward(ids, params = params)


    def _nucleus(self, logits: torch.Tensor, top_p: float, top_k: int):
        """Return list[(token_id, renormalized_prob)] after top-k then top-p."""
        logits = logits.detach().view(-1).float()
        probs = F.softmax(logits, dim = -1)
        k = min(top_k, probs.shape[-1])
        topk_probs, topk_idx = torch.topk(probs, k)
        cumsum = torch.cumsum(topk_probs, dim = 0)
        mask = cumsum <= top_p
        mask[0] = True
        over = (cumsum > top_p).nonzero(as_tuple = False)
        if over.numel() > 0:
            mask[over[0, 0]] = True
        kept_probs = topk_probs[mask]
        kept_idx = topk_idx[mask]
        kept_probs = kept_probs / kept_probs.sum()
        return list(zip(kept_idx.tolist(), kept_probs.tolist()))


    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(torch.tensor(ids, dtype=torch.long), decode_special_tokens = True)


    def _span_text(self, span_ids: list[int]) -> str:
        prefix = self._decode(self.tokens)
        full = self._decode(self.tokens + span_ids)
        return full[len(prefix):]


    def reset(self):
        self.tokens = []
        self.blocks = []
        self.cache_len = 0
        self.last_logits = None
        self.pending_branches = []


    def set_prompt(self, prompt: str):
        self.reset()
        self.prompt = prompt
        ids = self.tokenizer.encode(prompt, encode_special_tokens = True)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)  # [1, N]
        ids = ids.to(torch.long)
        n = ids.shape[1]
        if n == 0:
            raise ValueError("Empty prompt")
        if n > 1:
                params = {
                    "attn_mode": "flash_attn",
                    "cache": self.cache,
                    "past_len": 0,
                    "batch_shape": (1, self.cache.max_num_tokens),
                }
                self.model.prefill(ids[:, :-1], params = params)
        params = {
            "attn_mode": "flash_attn",
            "cache": self.cache,
            "past_len": n - 1,
            "batch_shape": (1, self.cache.max_num_tokens),
        }
        self.last_logits = self.model.forward(ids[:, -1:], params = params)
        self.cache_len = n
        self.tokens = ids[0].tolist()
        self.blocks.append({
            "tokens": self.tokens[:],
            "text": prompt,
            "is_prompt": True,
        })


    def rewind_to_block(self, block_index: int):
        if not (0 <= block_index < len(self.blocks)):
            raise ValueError("Invalid block index")
        self.blocks = self.blocks[: block_index + 1]
        new_tokens = [t for b in self.blocks for t in b["tokens"]]
        n = len(new_tokens)
        self.tokens = new_tokens
        self.last_logits = self._forward_one(new_tokens[-1], n - 1)
        self.cache_len = n
        self.pending_branches = []


    def commit_branch(self, branch_index: int):
        if not (0 <= branch_index < len(self.pending_branches)):
            raise ValueError("Invalid branch index")
        branch = self.pending_branches[branch_index]
        span = branch["tokens"]
        base = self.cache_len
        if len(span) > 1:
            params = {
                "attn_mode": "flash_attn",
                "cache": self.cache,
                "past_len": base,
                "batch_shape": (1, self.cache.max_num_tokens),
            }
            self.model.prefill(torch.tensor(span[:-1], dtype = torch.long).unsqueeze(0), params = params)
        self.last_logits = self._forward_one(span[-1], base + len(span) - 1)
        self.cache_len = base + len(span)
        self.tokens.extend(span)
        self.blocks.append({
            "tokens": span,
            "text": branch["text"],
            "is_prompt": False,
        })
        self.pending_branches = []


    def explore_branches(self, top_p: float, top_k: int, max_span: int):
        assert self.last_logits is not None, "No state; call set_prompt first"
        base = self.cache_len
        first_candidates = self._nucleus(self.last_logits, top_p, top_k)

        branches = []
        for tok_id, prob in first_candidates:
            span = [tok_id]
            if tok_id not in self.eos_token_ids:
                logits = self._forward_one(tok_id, base)
                pos = base + 1
                while len(span) < max_span:
                    next_cands = self._nucleus(logits, top_p, top_k)
                    if len(next_cands) != 1:
                        break
                    only = next_cands[0][0]
                    span.append(only)
                    if only in self.eos_token_ids:
                        break
                    logits = self._forward_one(only, pos)
                    pos += 1
            branches.append({
                "tokens": span,
                "text": self._span_text(span),
                "prob": float(prob),
            })
        self.pending_branches = branches
        return branches


session: Session | None = None


@app.get("/api/default_prompt")
def api_default_prompt():
    assert session is not None
    session.prompt = session.default_prompt()
    return {"prompt": session.prompt}


@app.get("/api/current_prompt")
def api_current_prompt():
    assert session is not None
    return {"prompt": session.prompt}


class InitReq(BaseModel):
    prompt: str
    top_p: float = 0.9
    top_k: int = 10
    max_span: int = 24


class CommitReq(BaseModel):
    branch_index: int
    top_p: float = 0.9
    top_k: int = 10
    max_span: int = 24


class RewindReq(BaseModel):
    block_index: int
    top_p: float = 0.9
    top_k: int = 10
    max_span: int = 24


def _state():
    assert session is not None
    return {
        "blocks": [
            {"text": b["text"], "is_prompt": b["is_prompt"]}
            for b in session.blocks
        ],
        "branches": [
            {"text": br["text"], "prob": br["prob"]}
            for br in session.pending_branches
        ],
    }


@app.post("/api/init")
def api_init(req: InitReq):
    try:
        session.set_prompt(req.prompt)
        session.explore_branches(req.top_p, req.top_k, req.max_span)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))
    return _state()


@app.post("/api/commit")
def api_commit(req: CommitReq):
    try:
        session.commit_branch(req.branch_index)
        session.explore_branches(req.top_p, req.top_k, req.max_span)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))
    return _state()


@app.post("/api/rewind")
def api_rewind(req: RewindReq):
    try:
        session.rewind_to_block(req.block_index)
        session.explore_branches(req.top_p, req.top_k, req.max_span)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))
    return _state()


@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "branch_decode" / "index.html",
        headers = {"Cache-Control": "no-store"},
    )


@app.get("/app.js")
def appjs():
    return FileResponse(
        STATIC_DIR / "branch_decode" / "app.js",
        media_type = "application/javascript",
        headers = {"Cache-Control": "no-store"},
    )


def main(args):
    global session
    model, config, cache, tokenizer = model_init.init(args)
    assert not model.caps.get("recurrent_states"), \
        "Recurrent models not currently supported in this demo."
    session = Session(model, config, cache, tokenizer)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True, default_cache_size = 8192)
    parser.add_argument("-host", "--host", default = "127.0.0.1")
    parser.add_argument("-port", "--port", type = int, default = 8000)
    _args = parser.parse_args()
    main(_args)
