"""Reference Duo DiT + uniform-state ancestral sampler. Dims: B batch, L canvas length, V vocab."""
import json

import torch
from safetensors.torch import save_file

from src import duo
from tests.conftest import EOS, TINY, tiny_dit

V = TINY["vocab_size"]


def test_forward_shape():
    dit = tiny_dit()
    x_BL = torch.randint(0, V, (3, 8))
    logits_BLV = dit(x_BL, torch.rand(3))
    assert logits_BLV.shape == (3, 8, V) and logits_BLV.dtype == torch.float32
    assert not torch.allclose(logits_BLV, dit(x_BL, torch.rand(3)))  # sigma conditioning reaches the output


def test_load_from_hf_layout(tmp_path):
    src = tiny_dit()
    state = {f"backbone.{k}": v for k, v in src.state_dict().items()}
    state["backbone.rotary_emb.inv_freq"] = torch.zeros(4)  # buffer present in the real checkpoint
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({**TINY, "model_length": 8, "causal": False}))
    dit = duo.load(str(tmp_path))
    for k, v in src.state_dict().items():
        assert torch.equal(dit.state_dict()[k], v)


def test_posterior_is_distribution():
    p_x0_BLV = torch.randn(2, 5, V).softmax(-1).double()
    x_BL = torch.randint(0, V, (2, 5))
    q_BLV = duo.posterior(p_x0_BLV, x_BL, alpha_s=torch.tensor(0.7), alpha_t=torch.tensor(0.4))
    assert q_BLV.shape == (2, 5, V) and (q_BLV >= 0).all()
    torch.testing.assert_close(q_BLV.sum(-1), torch.ones(2, 5, dtype=torch.float64))


def test_noise_removal_posterior_is_x0():
    # alpha_s = 1 (final step): the posterior collapses onto the x0 prediction wherever x_t carries no signal
    p_x0_BLV = torch.randn(1, 4, V).softmax(-1).double()
    x_BL = torch.randint(0, V, (1, 4))
    q_BLV = duo.posterior(p_x0_BLV, x_BL, alpha_s=torch.tensor(1.0), alpha_t=torch.tensor(1e-6))
    torch.testing.assert_close(q_BLV, p_x0_BLV, atol=1e-4, rtol=0)


def test_sample_shape_and_bos():
    x_BL = duo.sample(tiny_dit(), n=3, length=8, vocab_size=V, steps=4, bos_id=EOS)
    assert x_BL.shape == (3, 8) and x_BL.dtype == torch.long
    assert (x_BL[:, 0] == EOS).all() and x_BL.min() >= 0 and x_BL.max() < V
