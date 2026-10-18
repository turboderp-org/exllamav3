"""
ext.had_paley / ext.had_paley2 (hadamard.cpp, host code): fill a CPU fp16 (n, n) tensor with a Paley Hadamard matrix,
used by util.hadamard.get_hadamard for sizes the Sylvester doubling and the stored tables do not cover.

- had_paley, n = p + 1 with p prime, p = 3 (mod 4) (Paley I): H = I + [[0, 1^T], [-1, Q]] with
  Q[i, j] = legendre(i - j, p)
- had_paley2, n = 2 (p + 1) with p prime, p = 1 (mod 4) (Paley II): the conference matrix C = [[0, 1^T], [1, Q]]
  with each 0 replaced by [[1, -1], [-1, -1]] and each +-1 by +-[[1, 1], [1, -1]]
- both are Hadamard: entries +-1 and H H^T = n I (exact in float64)
- both reject non-fp16, non-2-D, non-square, non-contiguous and non-CPU tensors (they fill h from host code);
  had_paley2 also rejects n % 4 != 0 (it writes 2x2 blocks, and n = 2 would take residues mod 0)

Reference: an independent construction from the set of quadratic residues {x^2 mod p} (no modular exponentiation),
and the orthogonality identity itself.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.nogpu

PRIMES_3 = [3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83, 103, 107, 127, 131, 139, 251, 499, 1019, 2039]
PRIMES_1 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 113, 137, 193, 257, 509, 1009]


def _legendre(p: int) -> torch.Tensor:
    """(p,) chi(a) for a = 0..p-1"""
    chi = torch.full((p,), -1.0, dtype = torch.float64)
    chi[torch.tensor(sorted({(x * x) % p for x in range(1, p)}))] = 1.0
    chi[0] = 0.0
    return chi


def _q(p: int) -> torch.Tensor:
    i = torch.arange(p)
    return _legendre(p)[(i[:, None] - i[None, :]) % p]


def paley1_ref(p: int) -> torch.Tensor:
    n = p + 1
    s = torch.zeros(n, n, dtype = torch.float64)
    s[0, 1:] = 1.0
    s[1:, 0] = -1.0
    s[1:, 1:] = _q(p)
    return s + torch.eye(n, dtype = torch.float64)


def paley2_ref(p: int) -> torch.Tensor:
    m = p + 1
    c = torch.zeros(m, m, dtype = torch.float64)
    c[0, 1:] = 1.0
    c[1:, 0] = 1.0
    c[1:, 1:] = _q(p)
    a = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype = torch.float64)
    b = torch.tensor([[1.0, -1.0], [-1.0, -1.0]], dtype = torch.float64)
    return torch.kron(c, a) + torch.kron(torch.eye(m, dtype = torch.float64), b)


def _check_hadamard(h: torch.Tensor):
    n = h.shape[0]
    hd = h.double()
    assert ((hd == 1.0) | (hd == -1.0)).all()
    assert torch.equal(hd @ hd.T, n * torch.eye(n, dtype = torch.float64))


@pytest.mark.parametrize("p", PRIMES_3)
def test_paley1(p):
    h = torch.zeros((p + 1, p + 1), dtype = torch.half)
    ext.had_paley(h)
    assert torch.equal(h.double(), paley1_ref(p))
    _check_hadamard(h)


@pytest.mark.parametrize("p", PRIMES_1)
def test_paley2(p):
    n = 2 * (p + 1)
    h = torch.zeros((n, n), dtype = torch.half)
    ext.had_paley2(h)
    assert torch.equal(h.double(), paley2_ref(p))
    _check_hadamard(h)


def test_rejections(device):
    for f in (ext.had_paley, ext.had_paley2):
        with pytest.raises(RuntimeError):
            f(torch.zeros((12, 12), dtype = torch.float))
        with pytest.raises(RuntimeError):
            f(torch.zeros((12, 24), dtype = torch.half))
        with pytest.raises(RuntimeError, match = "must have 2 dimensions"):
            f(torch.zeros((12,), dtype = torch.half))
        with pytest.raises(RuntimeError, match = "must have 2 dimensions"):
            f(torch.zeros((12, 12, 2), dtype = torch.half))
        with pytest.raises(RuntimeError):
            f(torch.zeros((24, 24), dtype = torch.half)[::2, ::2])
        if device.type == "cuda" and torch.cuda.is_available():
            with pytest.raises(RuntimeError, match = "must be a CPU tensor"):
                f(torch.zeros((12, 12), dtype = torch.half, device = device))
    for n in (2, 6):
        with pytest.raises(RuntimeError, match = "multiple of 4"):
            ext.had_paley2(torch.zeros((n, n), dtype = torch.half))


def test_empty():
    # The 0 x 0 matrix: nothing to fill, no-op. Validation still applies
    for f in (ext.had_paley, ext.had_paley2):
        h = torch.zeros((0, 0), dtype = torch.half)
        f(h)
        assert h.shape == (0, 0)
        with pytest.raises(RuntimeError):
            f(torch.zeros((0, 0), dtype = torch.float))
