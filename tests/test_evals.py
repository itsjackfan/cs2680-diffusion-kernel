"""Sample-quality metrics. Dims: B batch, L sequence length."""
import math

import torch

from src.evals import entropy, gen_ppl


def test_entropy():
    tokens_BL = torch.tensor([[1, 1, 2, 2], [3, 3, 3, 3]])
    assert math.isclose(entropy(tokens_BL), math.log(2) / 2, rel_tol=1e-6)


def test_gen_ppl_finite(judge):
    tok, model = judge
    ppl = gen_ppl(["ABCDEFG", "HELLO", "XYZ"], tok, model, batch_size=2)
    assert math.isfinite(ppl) and ppl > 1
