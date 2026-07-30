import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.manual_seed(0)

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
V = 151936           # qwen-class vocab
B = 1024             # rows per launch (noise is independent per flat element)
N_LAUNCH = 1000
JUNK_LOGIT = -32.0
DEFICITS = [6.0, 9.0, 12.0]

logZ = math.log(1.0 + sum(math.exp(-d) for d in DEFICITS) + (V - 1 - len(DEFICITS)) * math.exp(JUNK_LOGIT))
draws = B * N_LAUNCH

def run(dtype):
    logits = torch.full((B, V), JUNK_LOGIT, dtype = dtype, device = device)
    logits[:, 0] = 0.0
    for i, d in enumerate(DEFICITS):
        logits[:, 1 + i] = -d
    noisy = torch.empty_like(logits)
    fn = ext.gumbel_noise_f16 if dtype == torch.half else ext.gumbel_noise_f32

    wins = torch.zeros(2 + len(DEFICITS), dtype = torch.long, device = device)
    max_noise = 0.0
    for i in range(N_LAUNCH):
        fn(logits, noisy, 0x1234567 + i * 7919)
        win = noisy.argmax(dim = -1)
        for k in range(1 + len(DEFICITS)):
            wins[k] += (win == k).sum()
        wins[-1] += (win > len(DEFICITS)).sum()
        if i < 25:
            max_noise = max(max_noise, (noisy.float() - logits.float()).max().item())

    print(f"\n== {str(dtype)}  ({draws} draws, max noise in first 25 launches: {max_noise:.2f})")
    ok = True
    for k, d in enumerate(DEFICITS):
        expect = math.exp(-d - logZ) * draws
        got = wins[1 + k].item()
        sigma = math.sqrt(expect)
        dev = abs(got - expect) / sigma if sigma > 0 else 0.0
        flag = "ok" if dev < 4.0 else "** OFF **"
        if dev >= 4.0: ok = False
        print(f"  deficit {d:5.1f}: expected {expect:9.1f}  observed {got:7d}  ({dev:.1f} sigma)  {flag}")
    junk = wins[-1].item()
    flat_floor = V * 3e-8 * draws
    print(f"  junk ({JUNK_LOGIT:.0f}):  expected ~0 (exact {V * math.exp(JUNK_LOGIT - logZ) * draws:.1e}; "
          f"the u=1.0 bug would give ~{flat_floor:.0f})  observed {junk}")
    if junk > 0: ok = False
    return ok

ok = run(torch.float) and run(torch.half)
print("\nRESULT:", "PASS" if ok else "FAIL")
