"""Reference Duo (s-sahoo/duo): DiT denoiser + uniform-state ancestral sampler, in plain PyTorch.
Ported from the checkpoint's model.py (no flash_attn) and github.com/s-sahoo/duo trainer_base.py. This is the numerical
oracle for the vLLM port (vllm/vllm/model_executor/models/duo.py): keep the math identical, never optimize it here.
Dims: B batch, L canvas length, V vocab, D hidden, H heads, K head dim, C cond dim, F sinusoid freqs."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return F.layer_norm(x.float(), x.shape[-1:]) * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, cond_dim, n_freq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_freq, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.n_freq = n_freq

    def forward(self, sigma_B):
        half = self.n_freq // 2
        freqs_F = torch.exp(-math.log(10000) * torch.arange(half, device=sigma_B.device) / half)
        args_BF = sigma_B[:, None].float() * freqs_F[None]
        return self.mlp(torch.cat([args_BF.cos(), args_BF.sin()], -1))


def rotary(x_BLHK, cos_LK, sin_LK):
    x1, x2 = x_BLHK.chunk(2, -1)
    return x_BLHK * cos_LK[:, None] + torch.cat([-x2, x1], -1) * sin_LK[:, None]


class Block(nn.Module):
    def __init__(self, dim, n_heads, cond_dim):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = LayerNorm(dim), LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)

    def forward(self, x_BLD, cos_LK, sin_LK, c_BC):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c_BC)[:, None].chunk(6, -1)
        h_BLD = self.norm1(x_BLD) * (1 + scale1) + shift1
        q, k, v = rearrange(self.attn_qkv(h_BLD), "b l (three h k) -> three b l h k", three=3, h=self.n_heads)
        q, k = rotary(q, cos_LK, sin_LK).to(v.dtype), rotary(k, cos_LK, sin_LK).to(v.dtype)
        a_BHLK = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x_BLD = x_BLD + gate1 * self.attn_out(rearrange(a_BHLK, "b h l k -> b l (h k)"))
        return x_BLD + gate2 * self.mlp(self.norm2(x_BLD) * (1 + scale2) + shift2)


class FinalLayer(nn.Module):
    def __init__(self, dim, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x_BLD, c_BC):
        shift, scale = self.adaLN_modulation(c_BC)[:, None].chunk(2, -1)
        return self.linear(self.norm_final(x_BLD) * (1 + scale) + shift)


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x_BL):
        return self.embedding[x_BL]


class DiT(nn.Module):
    """Duo backbone (state_dict keys = checkpoint keys minus `backbone.`). sigma = -log(alpha_t), one per sequence."""

    def __init__(self, vocab_size, hidden_dim, n_heads, n_blocks, cond_dim):
        super().__init__()
        self.vocab_embed = Embedding(vocab_size, hidden_dim)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.blocks = nn.ModuleList(Block(hidden_dim, n_heads, cond_dim) for _ in range(n_blocks))
        self.output_layer = FinalLayer(hidden_dim, vocab_size, cond_dim)
        head_dim = hidden_dim // n_heads
        self.register_buffer("inv_freq", 1.0 / 10000 ** (torch.arange(0, head_dim, 2).float() / head_dim), False)

    def forward(self, x_BL, sigma_B):
        x_BLD = self.vocab_embed(x_BL)
        c_BC = F.silu(self.sigma_map(sigma_B))
        freqs_LK = torch.outer(torch.arange(x_BL.shape[1], device=x_BL.device).float(), self.inv_freq).repeat(1, 2)
        cos_LK, sin_LK = freqs_LK.cos(), freqs_LK.sin()
        on_cuda = x_BL.device.type == "cuda"
        with torch.autocast(x_BL.device.type, dtype=torch.bfloat16, enabled=on_cuda):
            for block in self.blocks:
                x_BLD = block(x_BLD, cos_LK, sin_LK, c_BC)
            return self.output_layer(x_BLD, c_BC).float()


def load(hf_id):
    """hf_id: HF repo id or a local dir holding config.json + model.safetensors."""
    d = Path(hf_id) if Path(hf_id).is_dir() else Path(snapshot_download(hf_id))
    c = json.loads((d / "config.json").read_text())
    dit = DiT(c["vocab_size"], c["hidden_dim"], c["n_heads"], c["n_blocks"], c["cond_dim"])
    state = {k.removeprefix("backbone."): v for k, v in load_file(d / "model.safetensors").items()
             if k != "backbone.rotary_emb.inv_freq"}
    dit.load_state_dict(state)
    return dit.eval()


def alpha(t, noise_eps=1e-3):
    """Log-linear schedule: alpha_t = 1 - (1 - eps) t."""
    return 1 - (1 - noise_eps) * t


def posterior(p_x0_BLV, x_BL, alpha_s, alpha_t):
    """q(x_s | x_t, x0 ~ p_x0) for uniform-state diffusion (Duo eq. for 0 <= t-dt = s < t)."""
    V = p_x0_BLV.shape[-1]
    alpha_ts, d_alpha = alpha_t / alpha_s, alpha_s - alpha_t
    xt_BLV = F.one_hot(x_BL, V).to(p_x0_BLV.dtype)
    num = (alpha_t * V * p_x0_BLV * xt_BLV + (alpha_ts - alpha_t) * xt_BLV + d_alpha * p_x0_BLV
           + (1 - alpha_ts) * (1 - alpha_s) / V)
    return num / (alpha_t * V * p_x0_BLV.gather(-1, x_BL[..., None]) + 1 - alpha_t)


def sample_categorical(p_BLV):
    return (p_BLV / (1e-10 - (torch.rand_like(p_BLV) + 1e-10).log())).argmax(-1)


@torch.no_grad()
def ancestral_step(dit, x_BL, t, s, noise_eps):
    """One step from time t to s; s = 0 is the final noise-removal step (alpha_s = 1)."""
    alpha_t = alpha(torch.tensor(t, dtype=torch.float64, device=x_BL.device), noise_eps)
    alpha_s = alpha(torch.tensor(s, dtype=torch.float64, device=x_BL.device), noise_eps) if s > 0 else alpha_t ** 0
    sigma_B = (-alpha_t.log()).float().expand(x_BL.shape[0])
    p_x0_BLV = dit(x_BL, sigma_B).double().log_softmax(-1).exp()
    return sample_categorical(posterior(p_x0_BLV, x_BL, alpha_s, alpha_t))


@torch.no_grad()
def sample(dit, n, length, vocab_size, steps, bos_id=None, t_eps=1e-5, noise_eps=1e-3):
    """steps ancestral updates on linspace(1, t_eps) + one noise-removal step (= steps + 1 forwards).
    bos_id pins position 0 (the vLLM port serves a BOS prompt and pins canvas[0] to it, so both engines match)."""
    device = next(dit.parameters()).device
    x_BL = torch.randint(0, vocab_size, (n, length), device=device)
    ts = torch.linspace(1, t_eps, steps + 1).tolist() + [0.0]
    for t, s in zip(ts[:-1], ts[1:]):
        if bos_id is not None:
            x_BL[:, 0] = bos_id
        x_BL = ancestral_step(dit, x_BL, t, s, noise_eps)
    if bos_id is not None:
        x_BL[:, 0] = bos_id
    return x_BL
