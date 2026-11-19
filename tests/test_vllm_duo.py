"""vLLM Duo port (vllm/vllm/model_executor/models/duo.py) vs the reference src/duo.py, CPU, without a vLLM install:
the vLLM symbols duo.py imports are stubbed, the module is loaded by path. Engine wiring is covered by the smoke run.
Dims: B batch, L canvas length, V vocab, N flattened tokens."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src import duo
from tests.conftest import TINY, tiny_dit

PORT = Path(__file__).resolve().parents[1] / "vllm/vllm/model_executor/models/duo.py"
V, L = TINY["vocab_size"], 8


@pytest.fixture
def port(monkeypatch):
    stubs = {
        "vllm.config": dict(VllmConfig=object),
        "vllm.model_executor.layers.attention": dict(Attention=lambda *a, **kw: nn.Identity()),
        "vllm.utils.torch_utils": dict(async_tensor_h2d=lambda x, device, **kw: torch.as_tensor(x, device=device)),
        "vllm.v1.worker.gpu.model_states.default": dict(DefaultModelState=object),
        "vllm.v1.worker.gpu.sample.output": dict(SamplerOutput=SimpleNamespace),
    }
    for name, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    spec = importlib.util.spec_from_file_location("duo_port_under_test", PORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(port):
    ref = tiny_dit()
    hf = SimpleNamespace(hidden_size=TINY["hidden_dim"], num_attention_heads=TINY["n_heads"],
                         num_hidden_layers=TINY["n_blocks"], cond_dim=TINY["cond_dim"], vocab_size=V)
    model = port.DuoForDiffusion(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf),
                                                             cache_config=None))
    loaded = model.load_weights((f"backbone.{k}", v) for k, v in ref.state_dict().items())
    assert loaded == {k for k, _ in model.named_parameters()}
    return ref, model


def test_forward_matches_reference(port):
    ref, model = build(port)
    x_BL, sigma_B = torch.randint(0, V, (2, L)), torch.tensor([0.3, 2.0])
    # engine layout: a 1-token prefill (another request's BOS) then two canvases at positions 1..L
    input_ids = torch.cat([torch.tensor([5]), x_BL.flatten()])
    positions = torch.cat([torch.tensor([0]), torch.arange(1, L + 1).repeat(2)])
    sigma_N = torch.cat([torch.tensor([0.0]), sigma_B.repeat_interleave(L)])
    with torch.no_grad():
        logits_NV = model.compute_logits(model(input_ids, positions, sigma=sigma_N, seg_lens=[1, L, L]))
        torch.testing.assert_close(logits_NV[1:].view(2, L, V), ref(x_BL, sigma_B), atol=1e-4, rtol=1e-4)


def test_posterior_and_schedule_match_reference(port):
    p_x0 = torch.randn(2, 5, V).softmax(-1).double()
    x = torch.randint(0, V, (2, 5))
    a_s, a_t = torch.tensor(0.7, dtype=torch.float64), torch.tensor(0.4, dtype=torch.float64)
    torch.testing.assert_close(port.posterior(p_x0, x, a_s, a_t), duo.posterior(p_x0, x, a_s, a_t))
    T, t_eps, noise_eps = 4, 1e-5, 1e-3
    states = port.DuoRequestStates(1, L, V, T, 0, "cpu")
    ts = torch.linspace(1, t_eps, T + 1, dtype=torch.float64).tolist() + [0.0]
    for k in range(T + 1):
        states.step[0] = k
        sigma, alpha_s, alpha_t = states.times(torch.tensor([0]), t_eps, noise_eps)
        ref_s = duo.alpha(torch.tensor(ts[k + 1], dtype=torch.float64), noise_eps) if k < T else 1.0
        torch.testing.assert_close(alpha_t, duo.alpha(torch.tensor([ts[k]], dtype=torch.float64), noise_eps))
        torch.testing.assert_close(alpha_s.item(), float(ref_s))
