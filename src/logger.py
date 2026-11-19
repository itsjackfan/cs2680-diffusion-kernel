"""WandB + run_dir/metrics.jsonl. Keys must start with train/, test/ or perf/ (metric table: AGENTS.md)."""
from pathlib import Path
import json
import re

import wandb
from omegaconf import OmegaConf

PREFIXES = ("train", "test", "perf")


class Logger:
    def __init__(self, cfg):
        self.dir = Path(cfg.run_dir)
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode,
                   name=f"{cfg.git.branch}/{cfg.name}", group=cfg.git.branch,
                   id=re.sub(r"[^\w-]", "-", f"{cfg.git.branch}-{cfg.name}"), resume="allow",
                   dir=cfg.run_dir, config=OmegaConf.to_container(cfg, resolve=True))

    def log(self, metrics, step):
        bad = [k for k in metrics if k.split("/")[0] not in PREFIXES]
        if bad:
            raise ValueError(f"metric keys must start with one of {PREFIXES}: {bad}")
        out = {k: float(v) for k, v in metrics.items()}
        wandb.log(out, step=step)
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **out}) + "\n")

    def finish(self):
        wandb.finish()
