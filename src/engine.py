"""Decode engines: same interface, `generate(n, steps) -> tokens [n, L]` (canvas incl. pinned BOS at position 0).
HFEngine = plain-PyTorch reference (src/duo.py); VLLMEngine = the vLLM fork (third_party/vllm/, Duo port in
third_party/vllm/vllm/model_executor/models/duo.py). A new engine is a new class here + configs/engine/<x>.yaml."""
import torch

from src.duo import load as load_duo, sample


class HFEngine:
    def __init__(self, model, device, seed):
        torch.manual_seed(seed)
        self.model = model
        self.dit = load_duo(model.hf_id).to(device)

    def generate(self, n, steps):
        m = self.model
        return sample(self.dit, n, m.length, m.vocab_size, steps, m.bos_id, m.t_eps, m.noise_eps).cpu()


class VLLMEngine:
    """One BOS prompt token per request; the whole canvas is denoised for `steps` + 1 forwards and emitted at once
    (diffusion_read_only). Engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0, cluster/run.sbatch) so run.py
    times it directly."""

    def __init__(self, model, device, seed, llm):
        from vllm import LLM
        self.model = model
        self.llm = LLM(model=model.hf_id, tokenizer=model.tokenizer, seed=seed, max_model_len=model.length + 1,
                       max_num_batched_tokens=llm.max_num_seqs * (model.length + 1), **llm)

    def generate(self, n, steps):
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=self.model.length, ignore_eos=True, detokenize=False,
                                extra_args={"diffusion_read_only": True, "diffusion_max_steps": steps})
        outs = self.llm.generate([{"prompt_token_ids": [self.model.bos_id]}] * n, params, use_tqdm=False)
        return torch.tensor([o.outputs[0].token_ids for o in outs])
