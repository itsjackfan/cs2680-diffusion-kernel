"""One run: engine -> timed generation -> sample quality. Cluster entry: `python -m src.run <run_dir>/config.yaml`.
Dims: N samples, L canvas length."""
from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src import profiling
from src.evals import entropy, gen_ppl, load_judge
from src.logger import Logger


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main(cfg):
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    logger = Logger(cfg)
    ev, L = cfg.eval, cfg.model.length
    engine = instantiate(cfg.engine, model=cfg.model, device=cfg.device, seed=cfg.seed, _recursive_=False)

    engine.generate(ev.batch_size, ev.steps)  # warmup: autotune / lazy init stay out of the timings
    n_batches = ev.n_samples // ev.batch_size
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(cfg.device)
    t0 = time.perf_counter()
    tokens_NL = torch.cat([engine.generate(ev.batch_size, ev.steps) for _ in range(n_batches)])
    sync(cfg.device)
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if cfg.device == "cuda" else 0.0
    step_ms = 1000 * dt / (n_batches * (ev.steps + 1))

    prof = {}
    if "profile_steps" in ev:
        total_ms = profiling.region_times(lambda: engine.generate(ev.batch_size, ev.profile_steps), cfg.device,
                                          trace=str(run_dir / "trace.json"))
        region_ms = {r: total_ms[r] / (ev.profile_steps + 1) for r in profiling.REGIONS}
        prof = profiling.metrics(cfg.model, ev.batch_size, region_ms, step_ms, *profiling.peaks(cfg.device))

    tok, judge = load_judge(ev.judge, cfg.device)
    texts = tok.batch_decode(tokens_NL.clamp_max(tok.vocab_size - 1))  # Duo's extra id 50257 decodes as EOS
    with open(run_dir / "samples.jsonl", "w") as f:
        for toks, text in zip(tokens_NL.tolist(), texts):
            f.write(json.dumps({"tokens": toks, "text": text}) + "\n")

    n = len(tokens_NL)
    logger.log({
        "test/gen_ppl": gen_ppl(texts, tok, judge, ev.judge_batch_size),
        "test/entropy": entropy(tokens_NL),
        "perf/samples_per_s": n / dt,
        "perf/tokens_per_s": n * L / dt,
        "perf/latency_s": dt / n_batches,
        "perf/step_ms": step_ms,
        "perf/peak_mem_gb": peak_gb,
        **prof,
    }, step=0)
    logger.finish()


def cli(argv):
    main(OmegaConf.load(argv[0]))


if __name__ == "__main__":
    cli(sys.argv[1:])
