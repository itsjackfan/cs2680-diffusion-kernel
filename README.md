# cs2680-diffusion-kernel

Profiling and fused kernels for uniform-state diffusion LMs (Duo first), served through a vLLM fork, run on the
CS2680 cluster (SLURM), tracked in WandB. Proposal: [CS2680_Proposal.pdf](CS2680_Proposal.pdf).
**Read [AGENTS.md](AGENTS.md) first** (protocol, workflow, metrics, code idioms).

## Setup

Code is edited and committed locally only. `cluster/hooks/post-commit` rsyncs each worktree (incl. the `third_party/vllm/`
submodule checkout and `.env`) to `cs2680:~/cs2680-diffusion-kernel/<exp>/` on every commit; no git on the cluster.
Needs `Host cs2680` in `~/.ssh/config` and the Harvard VPN.

```bash
git submodule update --init                # third_party/vllm/ = fork at branch cs2680 (v0.31.0 + Duo port)
uv sync && uv run pytest -q                # local: Python 3.12 .venv, mocked CPU-only tests (no vLLM needed)
git config core.hooksPath cluster/hooks    # local, once per clone (shared by worktrees)
cp .env.example .env                       # fill in keys (git-ignored, synced)
curl -LsSf https://astral.sh/uv/install.sh | sh   # on cs2680, once
```

The first job in each synced `<exp>` dir builds `.venv` on its compute node (about 5 minutes); later jobs reuse it.

Pull results when needed: `rsync -az cs2680:cs2680-diffusion-kernel/<exp>/outputs/<exp>/<ablation>/ outputs/<exp>/<ablation>/`

## Run (humans only)

1. Work in a worktree: `git worktree add ../<exp> -b <exp>` (then `git -C ../<exp> submodule update --init`).
2. Add a line containing `` `<ablation>` `` to [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) saying what it tests. The launcher refuses unlogged names.
3. Commit (syncs), then launch on the login node:
   ```bash
   cd ~/cs2680-diffusion-kernel/<exp>
   alias launch='uv run --no-project --with hydra-core python cluster/launch.py'  # login node: hydra-only env
   launch <ablation> "<hydra overrides>"
   ```
   Comma values sweep (`engine=hf,vllm`), one sbatch job each.

Examples:
- Smoke, both engines (`SBATCH_TIMELIMIT` overrides `cluster/run.sbatch`):
  `SBATCH_TIMELIMIT=00:45:00 launch baseline-smoke "engine=hf,vllm eval.n_samples=16 eval.steps=64"`
- Baseline at Duo's 1024 steps: `launch duo-baseline "engine=hf,vllm"`

## Where things go

| What | Where |
| --- | --- |
| Run outputs (`config.yaml`, `freeze.txt`, `gpu.txt`, `slurm-*.out`, `metrics.jsonl`, `samples.jsonl`, `wandb/`) | `cs2680:~/cs2680-diffusion-kernel/<exp>/outputs/<exp>/<ablation>/` |
| Weights + judge (HF cache) | `$HF_HOME` |
| Metrics | WandB `jackfandev-harvard-university/cs2680-diffusion-kernel`, name `<exp>/<ablation>`, group `<exp>` |
| Reported tables/plots | `results/` in Git |

Delete smoke or failed runs as soon as you've processed them (run dir + WandB run).
