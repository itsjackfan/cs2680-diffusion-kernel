"""Dims: B batch, L canvas length, V vocab."""
from unittest.mock import MagicMock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.config import compose
from src.duo import DiT

TINY = dict(vocab_size=64, hidden_dim=32, n_heads=4, n_blocks=2, cond_dim=16)
EOS = 0  # stands in for GPT-2 <|endoftext|> (= Duo BOS) in the tiny vocab


def tiny_dit():
    torch.manual_seed(0)
    dit = DiT(**TINY)
    for p in dit.parameters():  # Duo zero-inits adaLN + output layer; randomize so outputs depend on inputs
        torch.nn.init.normal_(p, std=0.2)
    return dit


@pytest.fixture
def cfg(tmp_path):
    cfg = compose(["device=cpu", "wandb.mode=disabled", "engine=hf", "model.length=8",
                   f"model.vocab_size={TINY['vocab_size']}", f"model.bos_id={EOS}", "eval.n_samples=4",
                   "eval.batch_size=2", "eval.steps=3"])
    cfg.name, cfg.run_dir = "test", str(tmp_path / "run")
    cfg.git.branch, cfg.git.commit, cfg.git.dirty = "test-exp", "abc123", False
    return cfg


class FakeTokenizer:
    """Char-level stand-in for the GPT-2 tokenizer: text <-> ids mod vocab."""
    eos_token_id = EOS

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def batch_decode(self, tokens_BL):
        return ["".join(chr(65 + t % 26) for t in row) for row in tokens_BL.tolist()]

    def __call__(self, texts, **kw):
        ids = [[ord(c) % self.vocab_size for c in t] for t in texts]
        n = max(map(len, ids))
        return {"input_ids": torch.tensor([r + [EOS] * (n - len(r)) for r in ids]),
                "attention_mask": torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in ids])}


@pytest.fixture
def judge():
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=TINY["vocab_size"], n_positions=64, n_embd=16, n_layer=1, n_head=2))
    return FakeTokenizer(TINY["vocab_size"]), model.eval()


@pytest.fixture
def mocks(monkeypatch, judge):
    """No downloads, no WandB: tiny Duo for the HF engine, tiny judge for gen-ppl."""
    wandb = MagicMock()
    monkeypatch.setattr("src.logger.wandb", wandb)
    monkeypatch.setattr("src.engine.load_duo", lambda hf_id: tiny_dit())
    monkeypatch.setattr("src.run.load_judge", lambda name, device: judge)
    return wandb


def logged(wandb_mock):
    """All wandb.log calls as [(metrics, step)]."""
    return [(c.args[0], c.kwargs["step"]) for c in wandb_mock.log.call_args_list]
